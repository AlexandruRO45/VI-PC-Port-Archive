# Architecture

```
 user's files (outside repo)             VIctory
 ───────────────────────────             ─────────────────────────────────────────
 game assets ──────────────────────────► victory-pipeline + victory-game
                                         (victory-formats, victory-shader)
                                                    │
                                                    ▼
                                         cache_dir: converted data
                                                    │
 game executable ─► victory-loader ────► title code (native x86-64)
                    prepare, map, bind      │ imports
                                            ▼
                                         victory-hle ── host Windows (shared APIs)
                                            │ D3D12.X
                                            ▼
                                         victory-gpu ─► host D3D12
                                            │ shaders at pipeline creation
                                            ▼
                                         victory-shader ◄─► cache_dir
```

Shaders are translated when the title creates a pipeline, from the bytes it
passes, wherever it loaded them from (executable, archive, memory).
Converting them ahead of time from files is only a warm-up.

## Crates

| Crate | Role |
|-------|------|
| `victory-core` | config, asset-policy guard, logging |
| `victory-pe` | read-only PE32+ and import-library analysis |
| `victory-loader` | prepare, map, bind and start the title image; emulate CPU instructions the host lacks |
| `victory-hle` | console API implementations, host pass-through policy |
| `victory-gpu` | D3D12.X objects, command lists and GPU memory → host D3D12 |
| `victory-shader` | Xbox shader containers → host shaders |
| `victory-formats` | the GPU's texture layouts (sizes, tiling, detiling), audio codecs; planned: BCPack |
| `victory-pipeline` | converter trait, content-addressed cache |
| `victory-game` | Grand Theft Auto VI-specific formats, build identities, hooks |
| `apps/victory` | command-line front end |
| `xtask` | asset-policy check, git hooks |
| `tests/fixtures/hle-probe` | a stand-in "title" (our own code) for loader tests |

Everything except `victory-game` is platform-level and can be tested with
any Xbox Series title.

## Import resolution

For every `dll!function` the title imports, in order:

1. our implementation in `victory-hle::modules` (always wins);
2. the host's own function, if `HostPolicy` allows that DLL;
3. an export of a DLL the title ships beside its executable, mapped like
   the executable (only DLLs neither of the above serves);
4. a generated stub that names the import when called (`--unimplemented
   trap` breaks into the debugger and aborts; `zero` logs once and returns 0).

`LoadLibrary*` of a console-only DLL we implement (e.g. `xgameruntime.dll`)
returns a pseudo handle; `GetProcAddress` on it finds our functions.
`LoadLibrary*` of a title DLL returns its mapped image, mapping it first if
nothing imports it. By name, as by import, a title gets only host DLLs
`HostPolicy` passes through, and `GetProcAddress` on one returns our
override where import binding would.
`QueryApiImpl` hands out our runtime objects by provider and interface id
(`victory-hle::modules::xgameruntime` lists the areas); online areas answer
as an offline console. Game saves are files under `VICTORY_DATA_DIR`.

Title storage is a drive on the console, and titles build paths on that.
`victory-hle::modules::files` binds every path-taking title import
(kernel32 and the C/C++ runtime) and rewrites paths on a virtual drive to
the host folder behind it; persistent local storage is one such drive over
a folder under `VICTORY_DATA_DIR`. Other paths reach the host unchanged.

An exe's code reaches its static TLS as slot 0; mapped as a DLL it gets
another slot, so the loader redirects those loads (`victory-loader::tls`).
`victory run --execute` installs a crash reporter that names faults as
`module+rva` and unwinds the faulting thread's stack
(`victory-loader::crash`).

## Host CPU

Title code is built for the console's AMD Zen 2 and runs unchanged, so it
uses instructions only AMD CPUs implement. A host without them (every Intel
CPU) raises an illegal-instruction exception (`0xC000001D`) at each one.
`victory-loader::cpu` installs a vectored exception handler, first in the
chain, that carries the instruction out on the faulting thread's saved
registers and resumes after it; the crash reporter sees only what the
handler does not know.

| Instruction | Encoding | Emulation | In Grand Theft Auto VI |
|-------------|----------|-----------|------------------------|
| `CLZERO` | `0F 01 FC` | zeroes the 64-byte line holding `[rax]` | 11 sites; one routine that clears a 29 KB object runs a few times per frame |
| `INSERTQ` (SSE4a) | `F2 0F 78 /r ib ib`, `F2 0F 79 /r` | copies a bit field between the low halves of two XMM registers | 12 sites |
| `EXTRQ` (SSE4a) | `66 0F 78 /0 ib ib`, `66 0F 79 /r` | extracts a bit field of an XMM register | none found |
| `MONITORX`, `MWAITX` | `0F 01 FA`, `0F 01 FB` | no wait: `MWAITX` yields the thread and returns, as after a spurious wake-up | 2 pairs, in one wait loop |
| `RDPRU` | `0F 01 FD` | the time-stamp counter for both MPERF and APERF | 1 site |

The site counts are byte matches in the executable's code, read by eye
around each match; no disassembler has confirmed them. What a run really
executes is in its log: the handler names each instruction and its address
at its 1st, 2nd, 4th, 8th... use, and `victory run` prints the totals when
the title exits.

Design:

* **A handler, not a patch.** `CLZERO` is 3 bytes, too short for a jump, and
  finding the sites ahead of time would mean trusting a byte match in code.
  The handler acts only where the CPU itself refused an instruction, so it
  cannot hit data or the middle of another instruction, and a host that has
  the instruction never enters it (AMD hosts run everything natively).
* **Loops in one exception.** An exception costs microseconds. Compilers
  wrap `CLZERO` in two loop shapes (`victory-loader::cpu::clzero` lists
  them); the handler recognises them by the surrounding bytes and runs the
  whole loop, leaving registers and flags as the last pass would have. Any
  other use is carried out one instruction at a time.
* **Undefined results stay put.** SSE4a leaves the destination's high half
  undefined; the handler does not change it.

Limits:

* `MWAITX` never sleeps, so a title thread waiting with it spins through
  the handler (one exception and one yield per pass).

Measured on a Core i5-13600KF in the first minute of a run (boot and the
first screens, a steady 30 fps): `CLZERO` about 640 exceptions per second,
every loop taken whole; `MONITORX`/`MWAITX` one burst of about 16,000 pairs
in 30 ms, then none; `INSERTQ` under 1,024 uses, nearly all in one burst at
one site; `RDPRU` and `EXTRQ` not reached. An exception took about 1 µs.
The whole run (2 min 17 s, into the world) took 65,536 to 131,071 `CLZERO`
exceptions, 32,768 to 65,535 `MONITORX`/`MWAITX` pairs and 4,096 to 8,191
`INSERTQ`; still no `RDPRU` or `EXTRQ`.
* `RDPRU` reports a core that never changes frequency.
* `MOVNTSS`/`MOVNTSD` (SSE4a) are not emulated; the title does not contain
  them. Any other instruction the host lacks ends the title with a crash
  report naming `0xC000001D` and the address.
* A bad address in `rax` faults inside the handler instead of in title
  code; the report then names `victory.exe`.

`cargo test -p victory-loader cpu` executes the real instructions: through
the handler on a CPU that lacks them, natively on one that has them.

`victory census` shows this classification without running anything (it
does not map title DLLs; `victory run` without `--execute` does).

## Shaders

`victory_shader::ir::build` turns a native program into SSA, one
instruction per machine instruction. `victory_shader::structure` then
removes EXEC: lane masks become per-lane booleans, wave-wide mask tests
become per-lane branches, and vector instructions keep a guard only where
a lane can reach them with its EXEC bit clear. `victory_shader::resource`
traces each descriptor and scalar memory address to the root parameter it
comes from (table, heap, root descriptor or constant). `victory translate`
runs every shader it finds (containers, SGD5 files) through these and
lists what they cannot handle yet.
