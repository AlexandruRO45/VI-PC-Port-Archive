# Exe day: the first frame in five hours

**Status: prepared, not run against the retail executable.** The package used for the corpus does not contain
`game_scarlett_master.exe`, so this is the clock, the command block and the decision tree for the day it arrives.
[boot.md](boot.md) has the trace formats, the triage order and the Q1-Q22 list; this page is the timed plan that uses
them. Nothing here quotes game data: traces, captures and the overlay stay on the player's machine.

The target: executable at 17:00, a captured frame from the title's own present path by 22:00. If the delivered binary
lacks the capture route, the target is an offscreen PPM from the title's own draw; §1.1 gates that before 17:00. The
plan assumes the shader library is already staged, the build tree is warm and the first trace is read in order. It
spends 30 minutes proving the boot path, 90 on imports, 90 on the device and draws, an hour on pixels and the last 30
minutes on evidence.

**What is verified.** 26,503/26,503 shaders translate and pass `spirv-val`; the staged `<work>/shaders/spirv` +
`<work>/shaders/dxbc` layout resolves all 26,503 containers in `shader_library_corpus_test` with 0 failures
(`docs/design.md`, `runtime/README.md`). The delivered `gta6rt-final.exe` passes `check_tiny_frame.py` on a Vulkan
device. **What is not:** the retail executable has not been run, the PresentX queue slot is unmeasured, and the opt-in
`LEONIDA_PRESENTX_SLOT` / `LEONIDA_CAPTURE` route is in PR #6230 at `255efe094` and the final exe-day assembly, not on
main at `561b65a83`. The older `gta6rt.exe` intermediate does not contain those strings.

## 0. The build you are timing

Main at `561b65a83` does not contain all of the validated boot fixes. The kit's default is `gta6rt-final.exe`: base
`ba4b0bd4d` + #3777 + #5867/#5808/#5851 + #6230 at `255efe094` (the PresentX identity guard) + #6107/#6149 + the
runtime layer `7bce00bca` + upstream #6351 `XMemTransferMemory`. `gta6rt-final-pre-xmem.exe` is the same build minus
#6351. The older `gta6rt.exe` is the trace/draw intermediate. The kit's `MANIFEST.txt` describes the older
`exe-day-3777` build; read it for its test history, but use the three loaders and hashes in §1.1.

The final is the default because #6351 serves `XMemTransferMemory`, a Series startup call used by public
NetRumble/NativeAOT-Xbox bootstraps; without it the title can stop before the device. If you build main instead,
expect the fixed failures back, and check the build with the pre-flight tests rather than discovering the gap at
17:05.

The failures those fixes remove, so you recognise them if they return: an early `XErrorReport` stops the title; a
console-sized `CheckFeatureSupport(D3D12_OPTIONS)` returns `E_INVALIDARG`; a bundled module and a host DLL with the
same name collide; a mesh PSO is reported unsupported; a minimised or out-of-date present leaves a sticky `E_FAIL`.
None of them is a diagnosis to make during the five hours.

## 1. Pre-flight (before 17:00)

### 1.1 Use the delivered SDL3 binary

The kit at `C:\Users\kam\leonida-work\gta6rt-exeday` has three loaders. Prefer the newest:

| Loader | Bytes | SHA-256 | Capture route | Series startup |
|---|---|---|---|---|
| `gta6rt-final.exe` | 6,679,552 (A478) / 6,681,600 (0D32) | `A478A5DF41E3C79052B4118A8B272A3E30EC451C66834411A53A191FFD7E9E7B` / `0D32DD2CCCD9B2BE59DCCADCFC164F7492C4E6FEC80ADD8300B7B6E438B6E396` | yes | #6351 `XMemTransferMemory` |
| `gta6rt-final-pre-xmem.exe` | 6,672,896 | `A5F6124D6C8C43D5260B69C0A9AA4A5FDAC8EB8C6FF4520FDD3FB3FBD6D2BAD7` | yes | no #6351 |
| `gta6rt.exe` | 6,456,320 | `4F2C5156D0BAB04369E547C83D8809634DBAB233B8CDC052C697E68D1EF1ED16` | no | no #6351 |

`gta6rt-final.exe` is the default: SDL3 3.4.20 is static, the PresentX capture route is #6230 at `255efe094` (the
identity guard), and #6351 serves `XMemTransferMemory`. That last call is a Series startup call used by public
NetRumble/NativeAOT-Xbox bootstraps; without it the title can stop before the device. The handover build was
6,679,552 bytes (`A478`); the kit was refreshed while this page was being written to 6,681,600 bytes (`0D32`). Both
have all five strings and pass `check_tiny_frame`, so the selector accepts either.
`gta6rt-final-pre-xmem.exe` is the same build minus #6351, for a machine that must avoid it. The older `gta6rt.exe`
can trace and draw but has no present capture. Do not use the kit's `gta6rt-mesh-only.exe` or `gta6rt-nosdl3.exe`;
those are history.

Do not rebuild the loader just to run; a code fix in §3 is the exception. A source rebuild without
`-DLEONIDA_SDL3_FETCH=ON` omits the GameInput shim, and `gameinput.dll!GameInputCreate` becomes a boot blocker.

```powershell
$repo  = "C:\path\to\the-final-assembly-tree"   # not fresh main
$build = Join-Path $repo "runtime\build"
$work  = "E:\leonida-work"
$kit   = "C:\Users\kam\leonida-work\gta6rt-exeday"
$bin   = Join-Path $work "exeday\bin"
New-Item -ItemType Directory -Force $bin | Out-Null

$candidates = @(
    [pscustomobject]@{ Name = "gta6rt-final.exe"; Bytes = 6679552, 6681600; Sha = "A478A5DF41E3C79052B4118A8B272A3E30EC451C66834411A53A191FFD7E9E7B", "0D32DD2CCCD9B2BE59DCCADCFC164F7492C4E6FEC80ADD8300B7B6E438B6E396" },
    [pscustomobject]@{ Name = "gta6rt-final-pre-xmem.exe"; Bytes = 6672896; Sha = "A5F6124D6C8C43D5260B69C0A9AA4A5FDAC8EB8C6FF4520FDD3FB3FBD6D2BAD7" },
    [pscustomobject]@{ Name = "gta6rt.exe"; Bytes = 6456320; Sha = "4F2C5156D0BAB04369E547C83D8809634DBAB233B8CDC052C697E68D1EF1ED16" }
)
$selected = $candidates | Where-Object { Test-Path (Join-Path $kit $_.Name) } | Select-Object -First 1
if (-not $selected) { throw "no exe-day loader in $kit" }
$source = Join-Path $kit $selected.Name
$actualBytes = (Get-Item $source).Length
$actual = (Get-FileHash $source -Algorithm SHA256).Hash
if (@($selected.Bytes) -notcontains $actualBytes) { throw "$($selected.Name): $actualBytes bytes, expected $($selected.Bytes)" }
if (@($selected.Sha) -notcontains $actual) { throw "$($selected.Name): hash $actual, expected $($selected.Sha)" }
"selected $($selected.Name) ($actualBytes bytes, $actual)"

$files = $selected.Name, "gta6rt_tls_host.dll",
         "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"
$files | ForEach-Object { Copy-Item (Join-Path $kit $_) $bin -Force }
$loader = Join-Path $bin $selected.Name
Get-ChildItem $bin | Select-Object Name, Length
```

The exact check runs on `$loader`, the binary that will actually launch:

```powershell
$tokens = "LEONIDA_PRESENTX_SLOT", "LEONIDA_CAPTURE", "present-", "XMemTransferMemory", "SDL-release-3.4.20"
$ascii  = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($loader))
$tokens | ForEach-Object { "$_ = $($ascii.Contains($_))" }
$captureRoute = (@("LEONIDA_PRESENTX_SLOT", "LEONIDA_CAPTURE", "present-") |
                 Where-Object { -not $ascii.Contains($_) }).Count -eq 0
$xmemServed = $ascii.Contains("XMemTransferMemory")
$sdl3 = $ascii.Contains("SDL-release-3.4.20")
"capture route: $captureRoute; XMemTransferMemory: $xmemServed; SDL3: $sdl3"
```

`gta6rt-final.exe` has all five. `gta6rt-final-pre-xmem.exe` has the three capture strings and SDL3 but not
`XMemTransferMemory`; the older `gta6rt.exe` has only SDL3. The three capture strings are the gate for §5: with any
false, the binary can trace and draw but cannot write a frame artifact; plan the offscreen fallback in §5.4 and say
"trace/draw only" in the handoff. They are in PR #6230 at `255efe094` and the final assembly, not on `main` at
`561b65a83`. `XMemTransferMemory` is a separate boot check: if it is false, the title can stop before the device, so
use the final binary unless you have a reason not to.

If the delivered binary is lost, rebuild the tree that produced the selected one, with SDL3:

```powershell
cmake -S (Join-Path $repo "runtime") -B $build -G Ninja `
      -DCMAKE_BUILD_TYPE=Release -DLEONIDA_REQUIRE_VULKAN=ON -DLEONIDA_SDL3_FETCH=ON
cmake --build $build -j 8
```

A rebuild also needs the three CRT DLLs and `gta6rt_tls_host.dll` beside the new exe. `-DLEONIDA_REQUIRE_VULKAN=ON`
makes a missing Vulkan SDK a configure error instead of silently dropping the D3D12 device; if `cmake` or `ninja` is
not on `PATH`, use the copies under
`...\Microsoft Visual Studio\2022\...\Common7\IDE\CommonExtensions\Microsoft\CMake\`; the build script uses those.

Keep a last-good loader. All five files go together.

```powershell
$prev = Join-Path $repo "runtime\build-prev"
New-Item -ItemType Directory -Force $prev | Out-Null
Copy-Item (Join-Path $bin "*") $prev -Force
```

### 1.2 Stage the package, the movie and the shader library

`$game` is your own package root. Before import triage, check that the 16 modules the loader maps are present; a
missing DLL is a package-path problem, not a missing shim.

```powershell
$game = "D:\path\to\your\package"
$exe  = Join-Path $game "game_scarlett_master.exe"   # or wherever the exe was delivered

$modules = @(
    "Bink2Scarlett.dll", "XboxBvhBuilder.dll", "amd_fidelityfx_upscaler_xbox.dll",
    "amd_fidelityfx_loader_xbox.dll", "libHttpClient.GDK.dll", "XCurl.dll",
    "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vccorlib140.dll",
    "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"
)
$modules | ForEach-Object {
    $p = Join-Path $game $_
    [pscustomobject]@{ Module = $_; Present = Test-Path $p; Bytes = if (Test-Path $p) { (Get-Item $p).Length } else { 0 } }
} | Format-Table -AutoSize
```

All 16 must be `True`. `Bink2Scarlett.dll` is also the boot-movie decoder; the movie check is separate.

The title's boot movie is a likely first-frame path. `d3d12_boot_movie` decodes a real `.bk2` through that DLL and
presents it through PresentX on pixels. Point `LEONIDA_MOVIE` at a carved stream; if it is unset the test skips and
the movie path stays unverified. The stream stays on the player's machine.

```powershell
$env:LEONIDA_GAME_DIR = $game
$env:LEONIDA_MOVIE    = "E:\leonida-work\movies\boot.bk2"   # a carved Bink 2 stream
Test-Path $env:LEONIDA_MOVIE
```

The shader stage is the long pole: it carves 26,503 containers, disassembles them and translates every one. Start it
the day before. If `<work>\shaders\spirv\index.json` already exists, do not re-run it; verify it instead.

```powershell
python -I make_game.py --game $game --work $work --stages shaders
$env:LEONIDA_SHADERS = Join-Path $work "shaders\spirv"
Test-Path (Join-Path $env:LEONIDA_SHADERS "index.json")
(Get-ChildItem $env:LEONIDA_SHADERS -Filter 'VS_Bink_*' -File | Measure-Object).Count
```

`--stages shaders` needs the package; `--out` is not needed for this stage. The run needs about 30 GB for check,
shaders and runtime; the assets stage is ~370 GB and is not part of exe day. `LEONIDA_SHADERS` points at the `spirv`
folder, not at `shaders`; the corpus test reads `<spirv>/../dxbc` for the containers. The 1.0.2.1 corpus has four
`VS_Bink_*` modules; a zero count means the staged library is not the full corpus.

### 1.3 Verify the delivered binary, the corpus and the movie

`$tests` is the build tree that produced the selected binary; the kit's `MANIFEST.txt` describes the older
`exe-day-3777` build, so use the tree tied to `gta6rt-final.exe`. If it is gone, the delivered binary can still be
run, but these checks need a source build.

```powershell
$tests = "C:\path\to\the-final-build"   # the build that produced the selected exe
$env:LEONIDA_D3D12    = "vulkan"
$env:LEONIDA_SHADERS  = Join-Path $work "shaders\spirv"
$env:LEONIDA_GAME_DIR = $game

# 1. The exact delivered gta6rt-final.exe, through the synthetic tiny_frame guest.
python -I runtime\test\check_tiny_frame.py $loader `
    (Join-Path $tests "fixtures\game\Release\tiny_frame.exe") `
    --probe (Join-Path $tests "d3d12_com_test.exe")

# 2. The staged corpus (standalone; no gta6rt needed).
& (Join-Path $tests "shader_library_corpus_test.exe")

# 3. Present and movie core.
ctest --test-dir $tests -C Release `
    -R '^(d3d12_present|xbox_present|d3d12_boot_movie|bink_decode)$' --output-on-failure
```

CTest ignores `-C Release` on Ninja and needs it on the Visual Studio generator. If `shader_library_corpus_test.exe`
is not in `$tests`, build it with `cmake --build $tests --target shader_library_corpus_test`. Build
`d3d12_boot_movie` and `bink_decode` the same way if the kit's selection did not include them.

| Check | Pass looks like | If it does not |
|---|---|---|
| `check_tiny_frame.py` | `gta6rt-final.exe`: `OK`; the trace has device, queue, allocator, list, fence and the clear/copy; 32/32 pixels | no usable Vulkan device, a missing CRT DLL or a broken loader. Fix before 17:00. |
| `shader_library_corpus_test` | 26,503 containers resolve, 0 failures | `LEONIDA_SHADERS` is wrong or the staged layout is incomplete. |
| `d3d12_present`, `xbox_present` | `Passed`; the swapchain/present core and its capture work | no presentation-capable Vulkan device. The present hour is at risk. |
| `d3d12_boot_movie`, `bink_decode` | `Passed`; a real boot-movie frame is decoded and presented on pixels | `LEONIDA_GAME_DIR`/`LEONIDA_MOVIE` unset means SKIP; the movie path is unverified, not passed. |

A test missing from the build is a gap to record, not a pass. The corpus test reads every container; about 9 minutes
from a hard disk, up to an hour by CTest's timeout. Run it before 17:00. `check_tiny_frame` and the present tests take
seconds.

The corpus is 2,236 VS, 12,101 PS, 2,689 CS, 9,282 mesh and 195 GS. A mesh PSO is not an exotic case; it is most of
the corpus, and the mesh path is one of the branches in §0.

### 1.4 Freeze the command block

Every pass gets a new directory. `--trace` refuses an existing file (exit 2) rather than truncating it, so never
reuse a trace path.

```powershell
$stamp   = Get-Date -Format "MMdd-HHmmss"
$run     = Join-Path $work "exeday\$stamp"
$overlay = Join-Path $run "overlay"
$saves   = Join-Path $run "saves"
New-Item -ItemType Directory -Force $run, $overlay, $saves | Out-Null

$env:LEONIDA_D3D12              = "vulkan"
$env:LEONIDA_UNIMPLEMENTED      = "log"
$env:LEONIDA_SHADERS            = Join-Path $work "shaders\spirv"
$env:LEONIDA_GAME_DIR           = $game
$env:LEONIDA_SAVE_DIR           = $saves
$env:LEONIDA_VK_VALIDATION      = $null    # only while diagnosing a validation error
$env:DISABLE_RTSS_LAYER         = "1"      # only if Afterburner/RTSS is installed
$env:DISABLE_VULKAN_OBS_CAPTURE = "1"      # only if OBS is installed
```

Run every block from the repository root (`$repo`); the `python -I tools\...` and `make_game.py` paths assume it.

`LEONIDA_SAVE_DIR` is the offline `XGameSave` root (`<root>\local-user\<container>\<blob>`); a fresh folder per run
keeps saves out of the repository and makes a first-run save path visible. `$overlay` is where the title's writes go;
a fresh folder per run keeps the previous run's files from masking a failure.

`LEONIDA_D3D12=vulkan` is not optional: it selects the only backend that writes the `[d3d12-trace]` lines. Without it
the title runs through the PC D3D12 device and the D3D12 trace has nothing to read. `LEONIDA_VK_VALIDATION=1` adds the
Khronos validation layer, which is a diagnostic and not a timed-run setting.

### 1.5 The pre-flight checklist

```powershell
foreach ($d in @($overlay, $saves)) {
    New-Item -ItemType Directory -Force $d | Out-Null
    $probe = Join-Path $d ".write-probe"
    "ok" | Set-Content $probe
    Remove-Item $probe
}
```

| Before 17:00 | Done when |
|---|---|
| Loader | the selected loader's byte count and SHA-256 match §1.1; `$bin` has it plus `gta6rt_tls_host.dll` and the three CRT DLLs |
| SDL3/GameInput | `$sdl3` is true; if rebuilt, `-DLEONIDA_SDL3_FETCH=ON` was set and the GameInput shim is in it |
| Series startup | `$xmemServed` is true and `gta6rt-final.exe` is selected; `gta6rt-final-pre-xmem.exe` is only a deliberate fallback |
| Capture route | `$captureRoute` is true for a frame artifact; false means trace/draw only, not a frame artifact |
| Package | all 16 modules are present; `$exe` exists |
| Movie | `LEONIDA_MOVIE` is a file; `d3d12_boot_movie` passed (or the gap is recorded) |
| Shaders | `index.json` exists; four `VS_Bink_*` modules; corpus passed 26,503/26,503 |
| Device and present | `check_tiny_frame.py` passed on `gta6rt-final.exe`; the present tests passed |
| Overlay and saves | `$overlay` and `$saves` exist and the write probe succeeded |
| Traces | `$run` is empty; no path is reused between passes |
| Rollback | `runtime\build-prev` holds all five loader files |
| Clock | 17:00 start, 21:30 evidence freeze, 22:00 hard stop |

## 2. Hour 0-0:30 — trace

[boot.md](boot.md) §3 has the order: `map`, `unimplemented_call`, `tls_callback`/`dllmain`, `exception`, `exit`, then
the D3D12 trace. The first event that is not the expected one is the run's first fact.

| Clock | Do | Output |
|---|---|---|
| 0:00-0:03 | snapshot the build; set the environment; create `$run` | `$run\build.txt` |
| 0:03-0:10 | `--dry-run`; import list | `dry\startup.jsonl`, `dry\console.log`, `dry\imports.txt` |
| 0:10-0:17 | `--init` with both traces | `init\startup.jsonl`, `init\console.log` |
| 0:17-0:27 | `--run` with both traces; cap at 3 minutes | `run\startup.jsonl`, `run\console.log` |
| 0:27-0:30 | `boot_trace.py`; write the six facts | `run\summary.json`, `$run\triage.md` |

```powershell
git -C $repo rev-parse HEAD | Out-File (Join-Path $run "build.txt") -Encoding utf8
git -C $repo status --short | Out-File (Join-Path $run "build.txt") -Append -Encoding utf8
```

### 2.1 Dry run: map and imports

`--dry-run` maps the image and binds imports but runs no code, so it cannot produce D3D12 lines. Pass `--trace` only;
a `--trace-d3d12` log here would be empty and `boot_trace.py` would report it as an error.

```powershell
$dry = Join-Path $run "dry"
New-Item -ItemType Directory -Force $dry | Out-Null
& $loader --dry-run $exe --game-dir $game --overlay $overlay --trace (Join-Path $dry "startup.jsonl") 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File (Join-Path $dry "console.log") -Width 4096 -Encoding utf8
"dry exit=$LASTEXITCODE"

python -I tools\boot_trace.py (Join-Path $dry "startup.jsonl") --json |
    Out-File (Join-Path $dry "summary.json") -Encoding utf8
python -I tools\boot_trace.py (Join-Path $dry "startup.jsonl") --limit 20
```

Capture the module report and the import list:

```powershell
Select-String -Path (Join-Path $dry "console.log") -Pattern 'base=0x' | ForEach-Object { $_.Line }
Select-String -Path (Join-Path $dry "console.log") -Pattern '^\s+(shim|host|module|UNIMPLEMENTED)\s+' |
    ForEach-Object { $_.Line.Trim() } | Set-Content (Join-Path $dry "imports.txt") -Encoding utf8
(Get-Content (Join-Path $dry "imports.txt")).Count
Select-String -Path (Join-Path $dry "console.log") -Pattern 'unimplemented by return policy' |
    ForEach-Object { $_.Line }
```

`imports.txt` is the import list: one line per binding, with `UNIMPLEMENTED` marked. It names host APIs, not game
data, but it stays local with the rest of the run.

### 2.2 Init: TLS and DllMain

```powershell
$init = Join-Path $run "init"
New-Item -ItemType Directory -Force $init | Out-Null
& $loader --init $exe --game-dir $game --overlay $overlay --trace (Join-Path $init "startup.jsonl") --trace-d3d12 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File (Join-Path $init "console.log") -Width 4096 -Encoding utf8
"init exit=$LASTEXITCODE"

python -I tools\boot_trace.py (Join-Path $init "startup.jsonl") --d3d12 (Join-Path $init "console.log") --limit 20
```

`--init` runs every module's TLS callbacks and `DllMain(PROCESS_ATTACH)`. A failed DllMain here is cheaper to fix
than a crash at `--run`, and the summary names the module. If `--init` makes no D3D12 call, `boot_trace.py` prints
the startup report and an error that the D3D12 log has no trace line; that is expected, and the startup half is the
result.

### 2.3 Run: both traces

```powershell
$runDir = Join-Path $run "run"
New-Item -ItemType Directory -Force $runDir | Out-Null
& $loader --run $exe --game-dir $game --overlay $overlay --trace (Join-Path $runDir "startup.jsonl") --trace-d3d12 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File (Join-Path $runDir "console.log") -Width 4096 -Encoding utf8
"run exit=$LASTEXITCODE"

python -I tools\boot_trace.py (Join-Path $runDir "startup.jsonl") --d3d12 (Join-Path $runDir "console.log") --json |
    Out-File (Join-Path $runDir "summary.json") -Encoding utf8
```

If it has not exited after 3 minutes, stop it explicitly, unless the console log shows a PSO being created: the
largest shaders take 2-19 minutes to compile (`docs/design.md`, the full-corpus pipeline build), and a stall in
`CreateGraphicsPipelineState` is compile time, not a hang.

```powershell
Get-Process gta6rt -ErrorAction SilentlyContinue | Stop-Process
```

The startup trace is flushed per event, so it survives the kill; the last event names the wait. Do not rely on Ctrl+C:
PowerShell can leave the child alive behind it.

### 2.4 The six facts

```powershell
python -I tools\boot_trace.py (Join-Path $runDir "startup.jsonl") --d3d12 (Join-Path $runDir "console.log") --json |
    Out-File (Join-Path $runDir "summary.json") -Encoding utf8
python -I tools\boot_trace.py (Join-Path $runDir "startup.jsonl") --d3d12 (Join-Path $runDir "console.log") --limit 30
```

Before the first D3D12 call, the tool exits 2 after printing the startup report. The absent D3D12 half is the finding,
not a tool failure.

| Fact | Where | Record |
|---|---|---|
| Modules, relocation | dry summary `modules` | count; any `relocated: true` |
| First unimplemented | run summary `startup.unimplemented[0]` | `dll!name`, caller `module+RVA` |
| TLS/DllMain | run summary `startup.tls_callbacks`, `startup.dllmain` | counts; failed module |
| First exception | run summary `startup.exceptions[0]` | code, `at` |
| 24 device words | the `[rt]` line | all 24 hex words |
| First D3D12 call | run summary `d3d12.methods[0]` | method, outcome |

```powershell
$s = Get-Content (Join-Path $runDir "summary.json") -Raw | ConvertFrom-Json
$s.startup.unimplemented | Select-Object -First 1 | Format-List
$s.startup.dllmain | Format-List
$s.startup.exceptions | Select-Object -First 1 | Format-List
$s.d3d12.methods | Select-Object -First 12
Select-String -Path (Join-Path $runDir "console.log") -Pattern 'D3D12XboxCreateDevice params\[24\]' |
    ForEach-Object { $_.Line }
```

The 24 words are the title's device parameters. The shim logs them and does not use them to create the device; their
meaning is Q1. Copy the line into `$run\triage.md` so two runs can be compared. Do not decode it yet.

### 2.5 The 0:30 decision

| First fact | Next |
|---|---|
| A package DLL is missing, or the image is `relocated: true` | Fix the loader search path or the package layout; do not chase imports. |
| A `dllmain` is `0` | Fix that module first. Every later call from it is suspect. |
| An `exception` before the device | Loader, CRT or TLS. Use `LEONIDA_VK_VALIDATION=1` only if the stack points into the layer. |
| Unimplemented imports before the device | §3, one call at a time. |
| `D3D12XboxCreateDevice` reached | If the imports before it are benign, go to §4; otherwise finish §3 first. |
| No event after `trace_open` | The loader did not map or the trace path is wrong; check the exit code and stderr. |

Write the six facts and the decision in `$run\triage.md`. It is the handoff if the run ends at 22:00.

## 3. Hour 0:30-2:00 — boot

Goal: `--run` reaches `D3D12XboxCreateDevice` with no unimplemented call before it. `LEONIDA_UNIMPLEMENTED=log` is the
map; `stop` is the iteration tool.

First get the full unique list without stopping. Use a fresh directory and write the summary in the same block; the
`--trace` file must not exist yet.

```powershell
$allDir = Join-Path $run "all-imports"
New-Item -ItemType Directory -Force $allDir | Out-Null
$env:LEONIDA_UNIMPLEMENTED = "log"
& $loader --run $exe --game-dir $game --overlay $overlay `
    --trace (Join-Path $allDir "startup.jsonl") --trace-d3d12 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File (Join-Path $allDir "console.log") -Width 4096 -Encoding utf8
"all-imports exit=$LASTEXITCODE"

python -I tools\boot_trace.py (Join-Path $allDir "startup.jsonl") --d3d12 (Join-Path $allDir "console.log") --json |
    Out-File (Join-Path $allDir "summary.json") -Encoding utf8
$s = Get-Content (Join-Path $allDir "summary.json") -Raw | ConvertFrom-Json
$s.startup.unimplemented | Format-Table import, caller
```

Rank the list: calls from the executable before the device first, calls from a DllMain second, calls in a loop after
the device last. The last group does not block the first frame.

### 3.1 The iteration

Each iteration fixes exactly one call, rebuilds `gta6rt`, and reruns with `stop`.

```powershell
$env:LEONIDA_UNIMPLEMENTED = "stop"
if ($null -eq $i) { $i = 0 }
$i++   # 1, 2, ...; rerunning this block after a rebuild advances the iteration
$d = Join-Path $run ("boot-{0:d2}" -f $i)
New-Item -ItemType Directory -Force $d | Out-Null
& $loader --run $exe --game-dir $game --overlay $overlay --trace (Join-Path $d "startup.jsonl") --trace-d3d12 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File (Join-Path $d "console.log") -Width 4096 -Encoding utf8
$code = $LASTEXITCODE
"iteration $i exit=$code (3 = stopped at the first unimplemented call)"
Select-String -Path (Join-Path $d "console.log") -Pattern 'unimplemented:' | ForEach-Object { $_.Line }
```

Exit 3 means the `[rt] unimplemented:` line names the import, the caller as `module+RVA`, and rcx, rdx, r8 and r9 as
the title passed them. Exit 0 means the title exited before that call; read the trace's `exit` event and the D3D12
trace. A crash is an exception finding, not an import finding.

Fix, rebuild only `gta6rt`, then rerun the iteration block above; its `$i++` advances the iteration:

```powershell
cmake --build $build --target gta6rt -j 8
```

`$build` is the SDL3 build from §1.1 and `$repo` is the kit's source tree, not fresh main. If `$build` was never
configured, run the §1.1 fallback block first. A rebuilt `gta6rt.exe` needs the same three CRT DLLs and
`gta6rt_tls_host.dll` beside it; stage the set in a new folder and point `$loader` at it before rerunning.

### 3.2 What to do with the call

| Class | Where it goes | What to do |
|---|---|---|
| Host Windows API | `runtime/shims/registry.cpp` and a shim source | Add the binding and implement it; return the value the caller reads. |
| GDK family with an offline shim | `runtime/shims/xgameruntime_*.cpp` | Wire the call to the existing implementation before writing a new one. |
| HRESULT the title can survive | `LEONIDA_UNIMPLEMENTED=fail` | Let the title take its own failure path; implement only if it blocks. |
| Returns a pointer or struct the title reads | no zero return | Implement it or fail it. A zero return is a silent wrong value, not a shortcut. |
| Called every frame after the device | defer | Note it; the first frame does not need it. |

Record every fix in `$run\fixes.md`: iteration, time, `dll!name`, caller, decision, commit and the next first call.
That table is the only history that survives a rebuild.

### 3.3 The 2:00 stop

If the first unimplemented call is still before the device at 2:00, switch to `log`, take the full list and pick the
calls that gate the device. Do not spend the hour on `XStore` or `XGameSave` if the title can start offline; one run
with `fail` tells you whether it can. Two rules save the hour:

- Implement the call, the out parameter the caller reads, and the return value. Log the rest.
- Detach the debugger for the loop: `stop` breaks instead of exiting when a debugger is attached.

## 4. Hour 2:00-3:30 — device and draws

Goal: device, queue, allocator, list, fence, descriptor heap, first root signature, first PSO, first draw. The trace
order is the checklist.

```powershell
$d = Join-Path $run "device"
New-Item -ItemType Directory -Force $d | Out-Null
$env:LEONIDA_UNIMPLEMENTED = "log"
& $loader --run $exe --game-dir $game --overlay $overlay --trace (Join-Path $d "startup.jsonl") --trace-d3d12 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File (Join-Path $d "console.log") -Width 4096 -Encoding utf8

python -I tools\boot_trace.py (Join-Path $d "startup.jsonl") --d3d12 (Join-Path $d "console.log") --limit 40
Select-String -Path (Join-Path $d "console.log") -Pattern '^\[d3d12-trace\].* (failed|unsupported|refused|partial)$' |
    ForEach-Object { $_.Line }
```

The second command is the work queue: each line is one method and one outcome. Fix the first, rerun, repeat.

| Outcome | Meaning | Action |
|---|---|---|
| `failed` | the call returned a failure HRESULT | Read the `[rt]` line beside it; fix the argument, format or PSO. |
| `unsupported` | the layer knows it cannot do it | Implement it or route it to an existing path; check the console slot. |
| `refused` | a validation refusal | The layer rejected the call; the `[rt]` line names the reason. |
| `partial` | a submission ran some commands | The `[rt]` line names the omitted command; fix that, not the whole list. |

### 4.1 2:00-2:20 — device, queue, allocator, list, fence

The tiny_frame trace is the expected order: `D3D12XboxCreateDevice ok`, `CreateCommandQueue ok`,
`CreateCommandAllocator ok`, `CreateCommandList ok`, `CreateFence ok`, `CreateDescriptorHeap ok`.

- `D3D12XboxCreateDevice failed`: the HRESULT, the adapter, the IID and the 24 words are the first facts. The shim
  logs all of them. Do not decode the params; Q1 is still open.
- `CreateCommandQueue` `failed`, `unsupported` or `refused`: the device-slot assumption is wrong. The
  `[rt] device slot N` line names the slot. Record it; Q2.
- `CreateCommandList failed`: check the allocator and the list type. The PC list is not proxied yet; the layer's list
  path is what the tests cover.
- The trace stops after the device: the next call is the blocker, not the present.

### 4.2 2:20-2:50 — root signature and PSO

`CreateRootSignature` and `SetGraphicsRootSignature` come first. The console's `0x03000000` bits are masked by the
layer; if the PSO fails on the root signature, note the blob hash and move on. Q3 and Q19.

`CreateGraphicsPipelineState` or `CreatePipelineState` is the first big shader lookup. A failure usually means one of:
`LEONIDA_SHADERS` is wrong (the `[rt] d3d12: no translated shaders` line), the FSSM hash is not in `index.json`, the
stage split is wrong, or the PSO carries a subobject the layer does not model.

If the first PSO is a mesh PSO, the #1712 path must be in the build. On main it is not; the layer reports mesh shaders
as unsupported. Check the PSO stage before the draw. First PSO compilation can take minutes; a run that stops making
progress right after `CreateGraphicsPipelineState` is compiling, not hung. Give it 5-10 minutes before killing it.
`LEONIDA_VK_VALIDATION=1` is worth it here only if PSO creation returns a Vulkan validation error; it slows everything
else.

### 4.3 2:50-3:20 — the first draw

`SetPipelineState`, `SetGraphicsRootSignature`, `IASetVertexBuffers`/`IASetIndexBuffer`, `OMSetRenderTargets`, then
`DrawInstanced`, `DrawIndexedInstanced`, `Dispatch` or `DispatchMesh`. A clear-only frame counts as a first frame if
it comes from the title's own present path.

- `partial`: fix the named omitted command.
- `ok` but nothing appears: the present path or the render target is the next question, not the draw.
- `ExecuteIndirect` before the first draw: the rebased indirect path (`assist-3777-rebased`) is green but not on main
  at `561b65a83`; take it if the trace blocks there.

### 4.4 3:20-3:30 — choose the present path

```powershell
Select-String -Path (Join-Path $d "console.log") -Pattern 'PresentX|IDXGISwapChain|::Present|WaitFrameEventX|SetFrameIntervalX' |
    ForEach-Object { $_.Line }
```

An unknown queue or device slot has no method name in the trace: it appears as `SlotN`. Search
`XboxID3D12CommandQueue::Slot` and `[rt] d3d12_xs: queue slot` as well as the names above. Three outcomes: a queue
slot past the public chain (PresentX), a DXGI `Present`, or draws with no present call (offscreen fallback). Write
the decision in `$run\triage.md`.

## 5. Hour 3:30-4:30 — first pixels

Goal: a file on disk, non-black, produced by the title's own draw and present path. The trace's draw count must be
non-zero. A clear-only frame counts; a black frame does not. Check `$captureRoute` first: if it is false, this binary
cannot write a capture file, and the hour's deliverable is the offscreen PPM in §5.4, not a PresentX capture.

| Trace | Path | First move |
|---|---|---|
| `XboxID3D12CommandQueue::SlotN unsupported`, plus `[rt] ... queue slot N` | PresentX | Check `$captureRoute`; set slot N; capture. |
| `IDXGISwapChain::Present` or `CreateSwapChain` | DXGI | Keep the window visible; use the presenter capture. |
| Draws `ok`, no present call | offscreen | Copy the last render target to a readback buffer. |
| No draws | back to §4 | Fix the first draw; do not chase present. |

### 5.1 PresentX

GDKX samples present through PresentX and frame events and do not create a DXGI swapchain. That is a hypothesis for
this title, not a measurement. The queue proxy accepts the 19 public slots and logs a slot past them as
`[rt] d3d12_xs: queue slot N is beyond the PC queue vtable (19 slots); returning E_NOTIMPL`; the same call appears in
the D3D12 trace as `XboxID3D12CommandQueue::SlotN unsupported`. That N is the only evidence for the slot. Do not guess
30 or 32: a wrong slot can fault past the proxy's 128-entry table.

The gate is the exact binary, not the source tree: `$captureRoute` from §1.1. The flags are in PR #6230 at
`255efe094` and the final exe-day assembly, not on `main` at `561b65a83`; a `git grep` on main is expected to be empty
even when the selected binary has them.

```powershell
if (-not $captureRoute) {
    "this binary has no LEONIDA_PRESENTX_SLOT/LEONIDA_CAPTURE/present-: trace and draw only; use §5.4"
} else {
    $slotN = 30   # replace with the N the trace printed; do not guess
    $env:LEONIDA_PRESENTX_SLOT = "$slotN"
    $env:LEONIDA_CAPTURE       = Join-Path $run "capture"
    New-Item -ItemType Directory -Force $env:LEONIDA_CAPTURE | Out-Null
    # rerun the --run block with a fresh trace path
}
```

If `$captureRoute` is false, do not run the capture block; the binary cannot write the artifact. If it is true, the
branch's README is the authority for the slot value and the capture format. `$captureRoute` is necessary, not
sufficient: §5.2 still has to find and check the file.

After a run that names an unknown queue slot, `C:\Users\kam\leonida-work\discover-presentx-slot.ps1 -Log <console.log>
-Exe <exe> -GameDir <package>` counts the `XboxID3D12CommandQueue::SlotN` lines, ranks the candidates and prints the
relaunch with `-PresentXSlot N`. The printed command calls `launch-exeday.ps1`, whose `$bin` currently names the kit's
`gta6rt.exe` (the intermediate); point that at `gta6rt-final.exe` before using it, or run the final binary directly
with the same environment variables.

If the capture route writes raw RGBA instead of PPM or PNG, convert it in the route's source, not in this repository.
The capture is game data and stays local.

### 5.2 The non-black check

`Get-FrameStats` reads a binary PPM (P6) or a PNG, samples at most about 256 columns, and prints luminance statistics.
It needs `System.Drawing` for PNG, which is present in Windows PowerShell 5.1; if it is missing, convert the PNG to PPM
with ffmpeg and rerun.

```powershell
function Get-FrameStats {
    param([Parameter(Mandatory)][string]$Path)
    $bmp = $null; $step = 1
    if ([IO.Path]::GetExtension($Path) -eq '.ppm') {
        $b = [IO.File]::ReadAllBytes($Path)
        if ($b.Length -lt 11 -or $b[0] -ne 80 -or $b[1] -ne 54) { throw 'not a binary PPM (P6)' }
        $i = 2; $v = [System.Collections.Generic.List[int]]::new()
        while ($v.Count -lt 3) {
            while ($i -lt $b.Length -and ([char]$b[$i] -match '\s')) { $i++ }
            if ($i -lt $b.Length -and $b[$i] -eq 35) { while ($i -lt $b.Length -and $b[$i] -ne 10) { $i++ }; continue }
            $s = $i
            while ($i -lt $b.Length -and -not ([char]$b[$i] -match '\s')) { $i++ }
            $v.Add([int][Text.Encoding]::ASCII.GetString($b, $s, $i - $s))
        }
        if ($v[2] -ne 255) { throw 'only 8-bit PPM (maxval 255)' }
        $i++
        $w = $v[0]; $h = $v[1]
        $step = [Math]::Max(1, [int]($w / 256))
        $sample = {
            param($x, $y)
            $o = $i + 3 * ($y * $w + $x)
            if ($o + 2 -ge $b.Length) { throw 'truncated PPM' }
            @($b[$o], $b[$o + 1], $b[$o + 2])
        }
    } else {
        Add-Type -AssemblyName System.Drawing
        $bmp = [System.Drawing.Bitmap]::FromFile((Resolve-Path -LiteralPath $Path))
        $w = $bmp.Width; $h = $bmp.Height
        $step = [Math]::Max(1, [int]($w / 256))
        $sample = {
            param($x, $y)
            $c = $bmp.GetPixel($x, $y)
            @($c.R, $c.G, $c.B)
        }
    }
    $n = 0; $nonblack = 0; [long]$sum = 0; $min = 255; $max = 0
    for ($y = 0; $y -lt $h; $y += $step) {
        for ($x = 0; $x -lt $w; $x += $step) {
            $r, $g, $b2 = & $sample $x $y
            $l = [int](0.2126 * $r + 0.7152 * $g + 0.0722 * $b2)
            if ($l -gt 8) { $nonblack++ }
            $sum += $l; if ($l -lt $min) { $min = $l }; if ($l -gt $max) { $max = $l }
            $n++
        }
    }
    if ($bmp) { $bmp.Dispose() }
    if ($n -eq 0) { throw 'empty frame' }
    [pscustomobject]@{
        Width = $w; Height = $h; Sampled = $n
        NonBlackPct = [math]::Round(100.0 * $nonblack / $n, 2)
        MeanLuma = [math]::Round($sum / $n, 1); MinLuma = $min; MaxLuma = $max
    }
}
```

```powershell
if ($captureRoute) {
    $frame = Get-ChildItem -Path $env:LEONIDA_CAPTURE -Recurse -File -Include *.ppm,*.png |
             Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $frame) { throw 'no capture file; read the branch log for the reason' }
    Get-FrameStats $frame.FullName | Format-List
}
```

Pass: `NonBlackPct` above 0 and `MaxLuma` above 8. A non-black clear is a first frame for this runbook. Record `Width`,
`Height`, `NonBlackPct` and `MeanLuma` in `$run\triage.md`; do not attach the image to anything.

### 5.3 DXGI

The DXGI swapchain path is on main and `d3d12_present` passes on this machine. If the title calls
`IDXGISwapChain::Present`, use it. If it only calls PresentX, the DXGI path is not a drop-in fallback: there is no
swapchain to present through.

#5689 is the out-of-date/minimised present recovery. At `561b65a83` it is on `fix-dxgi-present-out-of-date`, not
main. If the build lacks it, keep the window visible and do not resize or minimise; an out-of-date present can
otherwise leave a sticky `E_FAIL`. Capture through the presenter's `enable_capture`/`read_back` if a capture switch is
exposed; otherwise a local screenshot is evidence for the handoff, not for the repository.

### 5.4 Offscreen

If draws complete but no present call is reached, copy the last render target to a `D3D12_HEAP_TYPE_READBACK` buffer
and dump PPM. The layer already has the resource and the copy path; the work is a hook at the last
`OMSetRenderTargets`/draw. Timebox it to 30 minutes. If it needs a new command or a new heap, stop and use the
capture-only PresentX route or the DXGI path. This proves the pixels, not the present path; say so in the handoff.

## 6. Hour 4:30-5:00 — evidence and fallbacks

21:30 is the evidence freeze. Stop changing code; a build that produced a frame is worth more than a fix that did not.

### 6.1 Evidence bundle (local)

```powershell
$last = $d   # or the directory of the last --run this session
$ev = Join-Path $run "evidence"
New-Item -ItemType Directory -Force $ev | Out-Null
Copy-Item -Path (Join-Path $last "startup.jsonl"), (Join-Path $last "console.log") -Destination $ev
python -I tools\boot_trace.py (Join-Path $last "startup.jsonl") --d3d12 (Join-Path $last "console.log") --limit 50 |
    Out-File (Join-Path $ev "triage.txt") -Encoding utf8
```

`triage.txt` can quote module names, import names, debug strings, resource names and paths. It stays on the player's
machine. The shareable summary is the next block. Record `$captureRoute` and the three token results beside it: a
binary without the capture route is a trace/draw result, not a first-frame artifact.

### 6.2 Shareable summary (counts and error classes only)

```powershell
$s = Get-Content (Join-Path $last "summary.json") -Raw | ConvertFrom-Json
$d3 = $s.d3d12
if ($null -eq $d3) {
    $d3 = [pscustomobject]@{ calls = 0; failed = @(); unsupported = @(); refused = @(); partial = @() }
}
[pscustomobject]@{
    events        = $s.startup.events
    modules       = @($s.startup.modules).Count
    tls_callbacks = $s.startup.tls_callbacks
    dllmain_ok    = $s.startup.dllmain.ok
    dllmain_fail  = $s.startup.dllmain.failed
    exceptions    = $s.startup.exception_count
    exit          = $s.startup.exit
    d3d12_calls   = $d3.calls
    failed        = @($d3.failed).Count
    unsupported   = @($d3.unsupported).Count
    refused       = @($d3.refused).Count
    partial       = @($d3.partial).Count
} | Format-List
```

This is the block to paste into an issue or the PR. It has no module names, no paths, no import names, no debug
strings, no shader names and no image. Public API method names are safe to add; their arguments are not.

If there is no frame, name the first blocker from the earliest `failed`, `unsupported`, `refused` or `partial` entry:

```powershell
$first = @($d3.failed + $d3.unsupported + $d3.refused + $d3.partial) |
         Sort-Object seq | Select-Object -First 1
$first | Format-List
```

Write it in `$run\blocker.txt` and in the handoff as `#seq method outcome`; do not paste the surrounding trace lines.

### 6.3 The three fastest fallbacks

| Fallback | Use when | Cost | What it proves |
|---|---|---|---|
| Offscreen readback | device and draws are `ok`; present is the only blocker | a 30-minute hook at the last render target | pixels from the title's draw, not the present path |
| DXGI path | the title creates a swapchain, or `IDXGISwapChain::Present` appears | minutes if the title uses it; not a drop-in if PresentX-only | the standard PC present path and the presenter capture |
| Capture-only PresentX | `$captureRoute` is true and the slot is known | minutes | the composed planes without a window; a first-frame artifact |

The order is deliberate: offscreen needs no unmeasured slot; DXGI is a real path only if the title uses it;
capture-only PresentX is last because the slot and the branch are the two unknowns.

### 6.4 Handoff

Five lines, by 22:00: build hash and branch, environment variables and `$captureRoute`, the first blocker, the last
good trace path, and the next three fixes. If `$captureRoute` is false, say "trace/draw only". Counts and error
classes only in the repository; the paths and the captures stay local.

## 7. Hard rules

These restate `AGENTS.md` for the pressure of the day.

- **No game data, ever.** No executable, archives, carved files, shaders in any form (DXBC, FSSM, disassembly,
  SPIR-V), textures, audio, or logs and reports that quote them in commits, issues, pull requests or fixtures.
- **Traces, captures, overlays and the import list stay on the player's machine.** `--trace` writes them; do not
  `git add` them; do not attach them to an issue. The startup trace can quote whatever the title prints.
- **Share counts and error classes.** Public D3D12/DXGI method names are fine. Arguments, resource names, debug
  strings, module paths, shader names and the captured image are not.
- **The overlay is the title's writes, not a save to commit.** Keep it out of the repository.
- **No keys, decryption or protection workarounds.** If a format turns out to be encrypted, write that down in
  `docs/design.md` and stop.
- **Nothing online.** No Xbox Live or Rockstar services or domains.
- **A test that needs a shader builds its input by hand or reads the player's work folder.** No captured shader goes
  into a fixture.

## 8. Verification for this page

Run on 2026-11-23 on Windows 11, NVIDIA GeForce RTX 5070 Ti (driver 621.24):

- Both `gta6rt-final.exe` builds (6,679,552 bytes / `A478A5DF41E3C79052B4118A8B272A3E30EC451C66834411A53A191FFD7E9E7B`
  and the 6,681,600-byte refresh / `0D32DD2CCCD9B2BE59DCCADCFC164F7492C4E6FEC80ADD8300B7B6E438B6E396`) passed
  `check_tiny_frame.py` against the synthetic `tiny_frame` guest:
  `tiny_frame under gta6rt --run on the Vulkan D3D12 device: OK`.
- The exact string scan is true for all five tokens on `gta6rt-final.exe`. `gta6rt-final-pre-xmem.exe` has the three
  capture strings and SDL3 but not `XMemTransferMemory`; the older `gta6rt.exe` has only SDL3.
- The staged 1.0.2.1 corpus: 26,503 containers, 26,503 reached, 0 not translated, 0 failures.
- A Release build of `561b65a83` produced `gta6rt.exe` and `gta6rt_tls_host.dll`; `run_tiny_frame`, `d3d12_present`
  and `xbox_present` passed on the source build (19 D3D12 calls, 0 failed/unsupported/refused, 32/32 pixels).
- `d3d12_boot_movie` and `bink_decode` were not run here because no `LEONIDA_MOVIE` was staged; the page treats that
  as a gap, not a pass.
- `boot_trace.py` passed its 20 unit tests and produced text and JSON reports from the synthetic `tiny_frame` trace.
- The 33 `test_docs_*` tests and the 9 `test_i18n_index` tests passed.
- The retail executable was not run.
