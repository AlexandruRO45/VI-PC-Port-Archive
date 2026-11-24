# VIctory — the Yeetbot159 fork

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files the
user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies the
platform underneath it and converts the title's data, ahead of time where it can
and on first use where it must (shaders). See [docs/architecture.md](docs/architecture.md).

**This repository contains no game files, code or assets.** You need your own
copy of the game. Nothing here is affiliated with or endorsed by Rockstar Games
or Take-Two Interactive, and nothing in the repository is derived from the
title's content: `docs/asset-policy.md` is the rule, `cargo xtask policy` is the
check, and `.githooks/pre-commit` runs that check on every commit once
`cargo xtask install-hooks` has pointed git at it.

## The three repositories this stands on

| Repository | What it is | Licence |
|---|---|---|
| [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (`origin`) | upstream: the loader, runtime services, Direct3D 12 back end and shader recompiler | GPL-3.0-or-later |
| [fxceless/victory-gta6](https://github.com/fxceless/victory-gta6) (`fxceless`) | the fork this one now follows as its base: the rendering and submission work that gets the world to draw | GPL-3.0-or-later |
| this repository (`fork`, Yeetbot159/VIctory) | one person's fork of the second, carrying its own fixes and instruments | GPL-3.0-or-later |

All three share upstream commit `ffe310e` as their common ancestor, so their
histories merge rather than being grafted: this fork's commit log contains
fxceless's own commits and upstream's, unsquashed. See
[Sources and attribution](#sources-and-attribution) for what came from where.

## Status, checked against `253eea0`

**Not playable.** The menus and the opening cutscene draw; a world frame has
never been captured. The base this fork adopted reports a cutscene and a
prologue on an NVIDIA card, which is why it was adopted, and on this PC the
adopted base now draws the cutscene too. Those are two different claims and the
table keeps them apart.

Every claim in this file carries one of three marks, and the file is re-checked
whenever `main` moves. At **`253eea0`** (2026-11-24), and re-checked at each move
(`b3b8533` → `214f670` → `082a618` → `253eea0`), with every verdict that changed saying so:

- **verified** — read in the code at that sha, or measured by this fork and
  reproduced from the artefact it rests on;
- **superseded** — a measurement of a tree that no longer exists in that form.
  It stays here so the record is not rewritten, but it is not a statement about
  this tree;
- **unverified** — quoted from somewhere the author of this file did not check.

| Stage | State | Produced by | Checked |
|---|---|---|---|
| Boot, first-boot screens, start screen, menus | works at the title's own 30 fps | this PC's card (AMD RX 6700 XT), upstream code, in the menus | unverified — upstream's own README |
| The cutscene, on the adopted base | the card presents it: 51 presents over 2,240 frames in a 180 s window, 0.0 % of captured pixels black, 27k–214k distinct colours per capture (means 118,115,45 at 512; 98,79,95 at 1024; 64,55,69 at 1280; 117,92,109 at 1300; 128,66,99 at 2048, the last a flat magenta fill) | the card, in the cutscene and menus, on the adopted base at `eb6d3a3` | measured at `eb6d3a3`, not re-measured after `b3b8533`, whose one new shader pass does not touch presentation |
| The cutscene window's depth target (frames 1,280–1,300) | the presented image changed frame to frame (24–144 distinct values in a 64x64 downscale, means 105,84,100 → 63,53,64 → 72,60,70 → 33,32,44 → 44,40,53, with 1,295 nearly flat) while the 3840x2160 `f19` depth target read **one unique value, zero** at all five | WARP, diagnostics build, keyed pad-A soak, offscreen, on the pre-adoption tree | **superseded** — measured at `0f564f8`. No run since has dumped a target, so this question is open on this base, not answered |
| The world, first frame, on our tree | every run ended here: access violation at `victory-image-…+0x8f92a`, read of `0x17`, rax all-ones — a descriptor slot holding `-1` used as a pointer — between frames 1,250 and 1,650 or not at all | this PC's card, in the world's first frame, on the pre-adoption tree | **superseded** — the adopted base has not reached a world frame in any window this crew has run, so the fault is neither reproduced nor disproved there |
| The world, the base's claim | the opening cutscene plays start to finish and the prologue renders characters, textures, sky, HUD and pause menu at **2–4 fps** | fxceless/victory-gta6, RTX 4070 12 GB / i7-14700KF, Series X profile, in the world | unverified — the base's own README |
| The world, this tree | **never captured**: the one card window on the adopted base ran 180 s with pad A every four seconds and never left the cutscene and the menus | the card, 2,240 frames, 0 device removals, 0 panics | verified — that window is the evidence, and it says nothing about the world |
| What a world-phase frame cost, on the adopted base | **7,379.8 ms**, of which textures were 6,950.0 ms and 6,896.6 ms of that was one wait for title memory to stop being remapped | a diagnostics build's frame log, read by the host lane | measured at `eb6d3a3`, in a world phase — [docs/host-cost.md](docs/host-cost.md) |
| Frame rate, on our tree | 30 fps in the menus; 0.5–1.8 s per world frame | this PC's card, in the world, before the world-frame crash | **superseded** — the pre-adoption tree |
| Frame rate, the base's numbers | 420–450 ms per frame by default; 240–350 ms with `VICTORY_RELAY=2` or with `VICTORY_LATE_WAIT=3 VICTORY_WIDE_BACK=1 VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_CARD_BUFFERS_MB=1024` | fxceless/victory-gta6, RTX 4070, in the world | unverified — the base's own README |

Every number here says what produced it: which tree, which adapter, and
whether the run was in the menus or in the world. Offline numbers — a
translation result, a validation count, a set of emitted DXIL — say that
instead, because no adapter touched them. A number that says none of this is a
number nobody can reproduce, and this file does not carry those.

Three statements this fork can stand behind today:

- **The cutscene draws on this PC, on the adopted base.** A 180 s window on the
  card, pad A every four seconds, presented 51 frames over 2,240 with no black
  pixels in any capture and 27,000 to 214,000 distinct colours in each —
  measured at `eb6d3a3`, in the cutscene and the menus. Whether that image is
  depth-tested is **open**: on the pre-adoption tree five dumped frames read
  zero in the 3840x2160 depth target at every one, and no run since has dumped a
  target.
- **No world frame has been captured.** The window above never left the
  cutscene. What the world phase does do is dispatch: once in the world the
  title asks for about 11,879 groups a frame, by its own summary line. The
  earlier "the world dispatches nothing" reading came from the menu phase and
  is wrong.
- **The frame rate belongs to a tree and a machine, and on the adopted base it
  belongs to one lock** — [What a frame costs](#what-a-frame-costs).

### The stages a run can reach, and what proves one

The goal this fork is now pointed at is not "the cutscene renders" but "the
cutscene finishes and the world is controllable", so a frame count on its own
says nothing. Every window is reported as one line — frames, the furthest stage
reached, the adapter, and the feature set the binary was built with:

```
frames=2,240 stage=cutscene adapter=card features=diagnostics
```

The stages, in the order the title reaches them:

```
boot screens → menus → profile (the account/picker button) → boot movie → cutscene → loading → world
```

Two things are true of that vocabulary today and both matter to whoever reads a
number next:

- **The layer emits no stage line.** `victory-hle`, `victory-gpu` and the app
  contain no code that reports which screen the title is on (a grep for one
  finds nothing), so a stage claim currently rests on a capture that has content
  at that point, plus the frame log around it. A run that sat in the menus for
  two thousand frames and a run that reached the cutscene produce the same kind
  of log; only the captures tell them apart.
- **One screen needs input to pass**: the profile picker, which pad A
  (`Enter`) answers, and which can ask twice. After it the title has advanced on
  its own in every window we have run — but no window has yet been long enough
  to tell a slow advance from a stall. The 180 s card window was still in the
  cutscene when its bound fired, and its log cannot say whether the title was
  working or waiting. `VICTORY_KEYS` schedules presses before the run starts;
  there is no live input channel in the tree yet, so a window that must press at
  a particular moment presses on a fixed schedule and hopes the prompt is late,
  which is why the schedule covers five minutes at three-to-five-second spacing
  rather than a few early presses.

### What this file did not check itself

Kept deliberately, because a claim you cannot reproduce is worth less than the
fact that you said so:

- The **545 → 0 mesh pipeline refusals with 338 modules** are the graphics lane's
  keyed soak as this crew's notes record it. Its log is not on the machine this
  was written on under any name that could be found, so it has not been
  recounted. The offline counts in the same table come from
  [docs/emit-gaps.md](docs/emit-gaps.md), which was measured.
- The **three `no host texture` failures a session** come from the review lane's
  handoff to whoever performed the merge, not from a log read here.
- The **base's** cutscene, prologue and frame-rate numbers are its own README's,
  on its own machine. Nothing in this repository reproduces them, and its report
  says so in its own words.
- The **world-frame crash** is described from this crew's crash logs on this PC,
  which are not committed because they name title addresses. The only
  `crashes.log` on disk while this was written belongs to the adoption attempt
  and carries different offsets, so the `+0x8f92a` signature is quoted from the
  crew's record rather than re-read.
- The **cutscene window's numbers** are this file's own: measured from the five
  dumped PNGs with a histogram, not quoted from anywhere.

The in-host instruments in [Debugging](#debugging) — `VICTORY_GPU_CHECK`,
`VICTORY_D3D12_VALIDATE`, `VICTORY_INDIRECT_RECORD`, `VICTORY_SERIAL_PIPELINES`,
the crumb ring and the frame log — are this fork's own work, and reach the tree
with the re-application of its fixes onto the adopted base.

## Known problems

- **Anything the sun does not hit directly is black** in the base's tree: the
  title's ray-traced bounce light and its lamp and sun shadows are not lit yet.
  The scaffold is in [docs/fork-ray-tracing.md](docs/fork-ray-tracing.md), is off
  by default, and by that document nothing is lit by it.
- **The first world frame crashed on our tree** — *superseded*: a descriptor
  slot held `-1` and the title dereferenced it, at a frame number that moved
  between 1,250 and 1,650 or did not come at all. On the adopted base no window
  has reached a world frame yet, so whether that fault survives the adoption is
  unknown rather than fixed.
- **Entering the world on a real GPU can hang the card**, and Windows then
  resets the display driver (every monitor goes black for a moment, and one
  2026-11-24 hang needed a restart). Use `VICTORY_D3D12_WARP=1` for anything
  long or unattended, and a wall-clock watchdog — 120 s, 180 s at most — for
  anything on the card.
- **The D3D12 debug layer, GPU-based validation and DRED are unavailable on the
  reference PC**: Windows' Graphics Tools optional feature is not installed, so
  `VICTORY_D3D12_DEBUG`, `VICTORY_D3D12_GBV` and `VICTORY_D3D12_DRED` report
  `no D3D12 debug layer` rather than doing anything. The same check was run again against the sibling's `1684989`, the clean-up pass over every
translated module, and against the merge that carried it: two rows below (the DirectStorage tail and the
descriptor stand-in) were found by that pass and were missing from the first version of this table. The in-host instruments in
  [Debugging](#debugging) exist because of that.
- **Zeroed measurements taken in the menus are not measurements of the world.**
  The title stops at an Xbox-style account picker and draws nothing 3D until a
  profile is chosen. A zeroed depth target read there has already misled one
  investigation.
- Around 24 GB of memory is in use during play in the base's tree.
- On this PC the Rockstar boot movie was reported at 15 fps with broken audio in
  one 2026-11-24 build, while our own baseline measured 30 fps with
  sample-identical audio at `b77a938`. *Superseded* for the adopted base, which
  presented the movie at 33–36 fps with one token per present, so the cadence
  there is not the defect; the 15 fps build was never bisected to a commit.

## Sources and attribution

### Marvisak/VIctory — GPL-3.0-or-later

The origin of everything structural: the loader that prepares, maps and binds
the title's image, the runtime services (`victory-hle`) that implement the
console's APIs, the Direct3D 12 back end (`victory-gpu`), and the shader
recompiler (`victory-shader`) that turns the console's GPU code into DXIL. Both
downstream forks keep upstream's history intact below their own commits.

### ivall/xcl — GPL-2.0-only

The project is now at [xcl-dev/xcl](https://github.com/xcl-dev/xcl);
`ivall/xcl` redirects there.

A reference for how this title behaves under another compatibility layer: it
runs console game code natively, reimplements the console's system libraries and
recompiles RDNA2 shaders to DXIL, which are the same problems this layer has.
Fixes it inspired were written anew in VIctory's code. **No code from XCL is in
this repository**, and none may be added: it is GPL-2.0-only, which does not
combine with this repository's GPL-3.0-or-later. That distinction is why this
section says "reference" rather than "source".

### fxceless/victory-gta6 — GPL-3.0-or-later

The base this fork now follows. Its commits are merged, not copied, so each one
can be read in the log. What came from it:

| Area | What it changed | Where |
|---|---|---|
| Rendering | meshlet culling read its scene buffers as the wrong kind of view, so the world drew almost nothing | `crates/victory-gpu/src/draw.rs`, `descriptor/` |
| Rendering | per-draw descriptor tables could run out, after which draws used slots never made for them and the card hung | `crates/victory-gpu/src/descriptor/host.rs` |
| Rendering | light packed by the game's compute shaders was read back through empty copies (sky, probes, fog), per mip and per layer; gathers at a named mip read level 0 | `crates/victory-gpu/src/shadow.rs` |
| Rendering | the material-program translations: fog reconstruction, shadow mask, hair, blend-shape skinning, fire, and the implicit-level samples that made compute modules invalid | `crates/victory-shader/src/emit/lower.rs`, `resource.rs` |
| Submission | submissions sent ahead of the card's wait (`VICTORY_LATE_WAIT`) with a relay thread landing results as the card finishes (`VICTORY_RELAY`) | `crates/victory-gpu/src/host.rs` |
| Submission | title memory opened as a graphics heap so buffers are used in place instead of copied (`VICTORY_INPLACE`) | `crates/victory-gpu/src/inplace.rs` |
| Ray tracing | the scaffold: builder programs translate, shader reads of title memory by raw address, and the hardware ray/box and ray/triangle test emulated in DXIL and checked against a Rust reference on the GPU | `crates/victory-shader/src/bvh.rs`, `emit/intersect.rs`, [docs/fork-ray-tracing.md](docs/fork-ray-tracing.md) |
| Stability | the hang the wave64 update introduced: its barrier form for 64-lane programs is off unless `VICTORY_WAVE_WIDE=1` | `crates/victory-shader/src/emit/wave.rs` |
| Stability | an intermittent device loss from the pipeline library file being replaced while mapped | `crates/victory-gpu/src/pipeline/library.rs` |
| Speed | clean-up passes over each translated module (`VICTORY_SHADER_OPT` bits: dead code, folding, reuse, branch facts, loads within a block, wave64 first-lane exchange), part of the shader cache key. Its author measures about 43 % fewer host instructions over the pack and about 20 ms less GPU time per frame on their machine | `crates/victory-shader/src/emit/bitcode/opt.rs` |
| Tooling | `translate` reports a census of the emitted code | `crates/victory-shader/src/emit/bitcode/census.rs`, `apps/victory/src/translate.rs` |

Its own README states the conditions on its numbers: one machine (Windows 11,
i7-14700KF, RTX 4070 12 GB, 32 GB RAM), Series X profile, and its commits were
developed with Claude Code.

### What this fork's own crew added

Each row is a change this fork wrote, with the file it lives in, the test that
pinned it, and the measurement that justifies it.

The third column is a **symbol check against the tree, on the sha named, never a
memory of what landed**: each verdict rests on a named symbol being present at
that sha or absent from it. A row marked **No** is a re-application that has not
happened yet, not a claim that the fix never worked — the measurement beside it
stands, on the tree that produced it. The adoption merge resolved the conflicted
GPU files in the base's favour, so six of the nine rows are no longer in the
tree. A row marked *carried by the base* kept its effect without our commit, and
*unverified* means nobody has checked whether the base's own form does the same
thing.

| Change | Where | In `253eea0`? | Measurement, with its conditions |
|---|---|---|---|
| An optimized clear value only on a texture that can take one | `crates/victory-gpu/src/resource.rs` (`host_texture_desc`) | **Yes, since the re-apply** — `host_texture_desc(desc, clear)` at `resource.rs:1631` now takes the clear value and returns the one that survives the flag rules (`(ResourceDesc, Option<[u32; 5]>)`); it was absent at `b3b8533` | A WARP probe: D3D12 answers `E_INVALIDARG` (`0x80070057`) for a clear value on a resource whose render-target flag the format's rules removed. Before the fix, three `no host texture` failures a session for NV12 (format 86) at 1024x4096 and B4G4R4A4_UNORM (115) at 2560x1440 |
| A view needs its resource's permission, and a clear it cannot serve is skipped | `crates/victory-gpu/src/host.rs` (`allows`, `clears_render_target`, `clears_depth`), `list.rs` (both clears) | **Yes, since the re-apply** — `clears_render_target` and `clears_depth` at `host.rs:45-54` both answer through `allows`; the once-only log helper `once_render` did not come back with them | The `op=clear_rtv` device removal at frame 1494, named by the breadcrumb ring (barrier, transition, resource_create, indirect, execute, dispatch, clear_rtv), is gone: a skipped clear records `b=1` instead of handing a null view to the runtime. Test: `a_depth_target_takes_no_unordered_view` |
| Mesh shaders declare no wave size | `crates/victory-shader/src/emit/lower.rs`, `emit/shader.rs` (the `PSV0` header) | **Carried by the base** — our test `a_mesh_shader_declares_no_wave_size` is absent, and mesh pipelines were refused 1 of 755 on the card and 0 of 69 on WARP at `eb6d3a3`, so the effect holds without our commit | In the title, WARP, one keyed soak, one commit apart: mesh pipelines refused **545 → 0** with **338** mesh modules translated in both runs. Offline, `victory translate <slice> --validate` ([docs/emit-gaps.md](docs/emit-gaps.md)): chunk `d010` emits 448 of 450 mesh programs and every emitted one validates, where none validated before; `d011` emits 427 of 427 and all validate. The two that do not emit stop at a resource-layer refusal, not a validation failure |
| Mesh output bounded by the program's declaration | `crates/victory-shader/src/emit/lower.rs` | **Carried by the base in its own form** — `max_vertices` is in `emit/container.rs`, `lower.rs`, `mod.rs` and `rect.rs`; our clamp and its test are not | The console's 10-bit `exp prim` indices reach 1023 while the program declares 128 vertices; a primitive naming a vertex past the declaration is culled, lanes past the declared outputs store nothing, and a group count past 65535 is clamped. Undefined behaviour on the card, so there is no failure count to quote |
| A scalar a pixel program spills through a VGPR lane | `crates/victory-shader/src/emit/lower.rs` (`v_readlane_b32` before cross-lane work) | **Unverified** — `v_readlane_b32` is in the base's emitter, and whether it resolves a constant-lane read the same way has not been checked | A pixel program refused an SGPR moved through a VGPR lane and read back at a constant lane as "a cross-lane operation outside compute" with nothing crossing lanes. The resolution is constant-lane only, phi arms must agree, and the write stays under the instruction's guard |
| Bindless pointer chasing, with its host half | `crates/victory-gpu/src/draw.rs`, `crates/victory-shader/src/emit/lower.rs`, `resource.rs` | **No** — `pointer_address` and the `checked_sub(256)` decode are still absent. The base solves the same problem its own way: an `& 15` runtime bias on an unaligned address (`draw.rs:694`, `1150`, `1222`) rather than a refusal, so the two designs are not interchangeable | The register encodes `((dword + 1) << 8) \| slot`, so the decode is `at >> 8` after `checked_sub(256)`; an address not on 16 bytes stays unbound rather than bound into an invalid call. Lane measurement (emitted DXIL validity, not a drawn frame): dump set A 13/24 → 19/29, set B 0/15 → 6/20 |
| A short DirectStorage stream fills its destination | `crates/victory-hle/src/modules/dstorage.rs` (`out[n..].fill(0)`, `window.bytes()[n..].fill(0)`, `short()`) | **No** — found by this audit and not earlier: the adopted tree's `inflate` returns the bytes written and leaves the rest as the title had it. The only true member of the `-1` family a host audit found: a stream shorter than its block left the tail holding `-1` and reported success. Re-apply pending, and it sits in the family the descriptor faults live in | A stream shorter than the block it is asked to fill, measured by reading the host copy back: tail `-1` and success before, zeroed after |
| A descriptor dword nothing reads as a number keeps a stand-in | `crates/victory-shader/src/emit/lower.rs` (the dword walk before the program is refused) | **No** — the adopted tree refuses such programs instead (`a descriptor's address or format read as data`). That is a translation difference, not a behaviour one: our tree emitted programs the base refuses, so the base's 324 refusals over the corpus include this family | Programs whose only reader of a dword is a resource op that already holds a host handle translate here and are refused there |
| Remembered buffer state on the shadow | `crates/victory-gpu/src/shadow.rs` (`state`), `indirect.rs`, `list.rs` | **No** — no `Shadow` state under that name, and `indirect.rs` names `INDIRECT_ARGUMENT` in comments only; re-apply pending | A barrier naming a before-state the buffer was not in. An explicit transition does not decay, so the indirect buffer's UNORDERED_ACCESS and the copy's COPY_DEST are states a later barrier must leave from. Found by review, pinned by `clamp_preserves_record_offsets_and_shared_budget`; no run measurement |
| The instrument set | `host.rs` (`VICTORY_GPU_CHECK`, `VICTORY_D3D12_VALIDATE`, the crumb ring, the frame log), `draw.rs` (`VICTORY_INDIRECT_RECORD`), `pipeline/host.rs` (`VICTORY_SERIAL_PIPELINES`, retry diagnostics) | **Mostly, since the re-apply** — the crumb ring, the frame log, `VICTORY_INDIRECT_RECORD` and `VICTORY_SERIAL_PIPELINES` are in; `VICTORY_GPU_CHECK` and `VICTORY_D3D12_VALIDATE` are still absent, and they are the two that name a removal at the operation and at the submission | What turned "the application made an invalid call" into `dispatch 1x1x1 [compute e30970f1727063fd]`, then into `submission=31371 list=0 of=1`, then into `op=resource_create a=0xb6456ee00 b=3 (invalid call, 0x887A0001)`. Each switch is in [Debugging](#debugging) with its cost |
| The test gate | `host.rs` (`the_device_tests_have_a_device`), `shadow.rs` per-copy accounting, `frame.rs` (`waits_end_on_ticks_and_count_frames`) | **Partly** — `waits_end_on_ticks_and_count_frames` exists (the base has its own) and the copy accounting came back as `kept_most()` in `shadow.rs`; `the_device_tests_have_a_device` does not exist, so a `victory-gpu` suite can still go green having asserted nothing on a machine with no device | A green `victory-gpu` suite where every device test early-returned is a suite that asserted nothing. The other two measured the machine rather than the code: one flip per tick against a 1.67 ms clock (4 of 20 parallel runs failed on unmodified `main`), and a process-global registry every other test writes (1 of 20) |

Two more are written and measured but **not in this tree yet**, so no row above
claims them:

- **Implicit-level samples outside the pixel stage.** `image_sample`,
  `_o`, `_cl` and `_b` mapped to DXIL opcode 60/61 in every stage; outside a
  pixel shader there are no neighbouring fragments, so `dxil.dll` refuses the
  module with "features incompatible with the shader stage" and no pipeline is
  created for it. Mapping them to a level-zero `sampleLevel` outside pixel
  stages makes `015832d72477fb26` — one of the modules no run had ever bound —
  validate and create. The change is uncommitted in the shader lane's worktree
  as of this writing.
- **A session guard.** One title session per machine, held as a named object by
  the operating system rather than by a lock file, so a second launch leaves
  before it touches the GPU and a child that outlives its parent still holds the
  exclusion (`apps/victory/src/session.rs`, `VICTORY_SESSION_INHERITED`). Open
  as a pull request on its own branch, not merged.

## Building and running

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 (and mesh shaders for the world), and Rust, pinned in
`rust-toolchain.toml` (1.98.1, `x86_64-pc-windows-msvc`).

The title also needs several GiB of available Windows commit at startup — check
**Committed** under **Task Manager → Performance → Memory**. If `crashes.log`
reports Windows error 1455, close other applications or enlarge the paging file
and retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo build --release -p victory
cargo run -p victory -- config           # check the paths you just set
```

Point `victory.toml` at your own copy of the game, outside the repository — the
tools refuse a path inside it. `VICTORY_GAME_DIR`, `VICTORY_CACHE_DIR`,
`VICTORY_DATA_DIR` and `VICTORY_TEST_TITLES` override the file, and
`VICTORY_CONFIG` selects a different file. Derived data goes to
`%LOCALAPPDATA%\victory\cache` and the title's saves, settings and
`crashes.log` to `%LOCALAPPDATA%\victory\data` (the log is under
`<data>\<build id>\temp\`).

Run the title, and give it a bound:

```powershell
target\release\victory.exe run "<game dir>\<the title's exe>" --execute
```

`run` without `--execute` binds the imports and stops, which is the cheap check
that your files and paths are right. `--unimplemented trap|zero` decides what an
unimplemented import does when called, `--strict` disables host pass-through,
`--console series-x|series-s` picks the profile the title sees, and everything
after `--` goes to the title's command line. The window caption shows the
framerate.

Tests need no game data and run in CI (`.github/workflows/ci.yml`: clippy with
`-D warnings` on both feature sets, then the suite):

```powershell
cargo test --workspace
cargo test --workspace -- --test-threads=1   # the result that gates a merge
```

Two assertions in `victory-gpu` have measured the clock and a process-global
registry rather than the code (`frame::tests::waits_end_on_ticks_and_count_frames`
and the shadow snapshot tests). Both were rewritten in this fork, and the base
carries its own form of each — its `frame.rs` has a test of that name too — so
the serial run is what a merge decision reads whichever version is in the tree:
a compile error prints neither `FAILED` nor a `test result:` line, so green means
a `test result: ok` you read for every crate, never the absence of a failure.

Input is XInput pads. With none connected, the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

Offline subcommands, none of which start the title or need a GPU session:

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory config
```

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

The switches are grouped by what you would reach for. Everything marked
*diagnostics* exists only in that build. Two rows are marked **not in this
tree**: `VICTORY_GPU_CHECK` and `VICTORY_D3D12_VALIDATE` are this fork's own
instruments, written and used, which the adoption merge has not restored —
`VICTORY_SERIAL_PIPELINES` was in the same state at `214f670` and came back with
the re-apply. They describe what the re-apply still owes, not switches you can
set against `253eea0` today. The base carries about a
hundred more of its own, mostly experiments, each described in a comment where
it is read; its README is the index for those.

### What is on screen

| Variable | Effect |
|---|---|
| `VICTORY_CAPTURE=<dir>` | *diagnostics.* Saves each presented frame whose number is a power of two as `frame-<n>.png`, at half size |
| `VICTORY_CAPTURE_FRAMES=<n>,<n>` | *diagnostics.* Additional frames to save, and their draws traced at debug level |
| `VICTORY_DUMP_TARGETS=1` | *diagnostics.* With a capture folder: every texture the listed `VICTORY_CAPTURE_FRAMES` frames render to or write, saved at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`. Floats are scaled by their largest value and depth by its range, so structure shows. Only the listed frames — a power-of-two frame alone dumps nothing |

### What the GPU did

| Variable | Effect |
|---|---|
| `VICTORY_GPU_CHECK=1` | **Not in this tree** (re-apply pending). *diagnostics.* Asks the device after every recorded host operation whether it has been removed, so the first operation after which it has names itself: `the device was removed right after op=clear_rtv a=… b=…`. One device query per operation, so it is very slow — never compare frame times with it on |
| `VICTORY_D3D12_VALIDATE=1` | **Not in this tree** (re-apply pending). One submission per command list, each asked for the removal reason as it lands; the first that comes back removed is logged as the submission that removed the device with that list's newest operations. It removes the overlap between lists, so a removal needing two lists in flight will not reproduce under it: take the list it names as guilty, not the whole story |
| `VICTORY_INDIRECT_RECORD=1` | *diagnostics.* Logs an indirect execution's record: kind, stride, count, the words the title left, the thread count or draw count they name, and whether the range fits the buffer. The first 64 executions and then every 256th, so the log stays readable |
| `VICTORY_GPU_TIMING=1` | *diagnostics.* Runs and times each list of a submission on its own, and logs slow ones with their slowest commands |
| `VICTORY_SKIP_PIPELINES=<hash>,…` | *diagnostics.* Omits draws and dispatches, indirect ones included, whose pipeline contains a listed 16-digit shader hash — the way to subtract a suspect pipeline and confirm cause rather than neighbour |
| `VICTORY_D3D12_DRED=1` | *diagnostics.* Enables device-removed data: the allocation a fault hit and the commands in flight |
| `VICTORY_D3D12_DEBUG=1` | *diagnostics.* The host debug layer (needs Windows' Graphics Tools) |
| `VICTORY_D3D12_GBV=1` | *diagnostics.* With the debug layer: GPU-based validation without state tracking, so out-of-range descriptor indices become messages naming the command instead of page faults. Slow |
| `VICTORY_INDIRECT_RECORD=1` | logs an indirect execution's record: kind, stride, count, the words, the thread count they name, and whether the range fits the buffer. The first 64 and then every 256th. Reads the host copy as uploaded; the clamp's own counters, not this log, are the trustworthy group count |
| `VICTORY_NO_ARGS_REFRESH=1` | **A probe, never an option, and unreachable from a release build** — `host.rs` returns false without the `diagnostics` feature, so a variable inherited from a wrapper cannot switch it on: skips the upload of the host copy over a title argument buffer before an indirect execution. That copy is the only path a title-written record has to the GPU, so with it off every mesh dispatch asks for zero groups (measured: 102 of 102 summaries at zero, against 11,879 groups a frame with it on). It answered whether we overwrite records the title wrote: we do not |
| `VICTORY_D3D12_WARP=1` | *diagnostics.* The host device on the software adapter: a shader's wild access becomes an access violation in this process instead of a GPU fault and driver reset. Slow, and it cannot black the machine, which is why soaks and unattended runs use it |

### What the shaders did

| Variable | Effect |
|---|---|
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | *diagnostics.* Translated loops stop after `n` iterations per lane (decimal or `0x` hex, at least 1) instead of 2^20. The value is part of the translation, so it takes its own cache entries. If a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_SKIP_PIPELINES=<hash>,…` | as above: omits work whose pipeline contains that shader |
| `VICTORY_SERIAL_PIPELINES=1` | Creates host pipelines one at a time, through one lock that every path into the vendor's compiler takes (the lazy helpers included), to tell a driver race from a description we build differently each run. A refusal that a later attempt at the same module survives is not a verdict on the module |

### What the process did

| Variable | Effect |
|---|---|
| `VICTORY_PASS_COST=1` | charges each submission's wall time to the pipelines its lists bound and logs the worst six once a second of accounted time, with each one's share of the accounted total. It also logs the wall clock between submissions and the percentage those passes account for — when that remainder is large it is the finding, not the ranking |
| `VICTORY_LOG_FLUSH=1` | Flushes after every event, so a log cut short by a kill or a device loss ends at its real last event instead of at a buffer boundary — which is where the operations just before a removal used to disappear. Costs a write per line. No wrapper in this crew's run harness sets it, so set it yourself on any run whose log matters |
| `VICTORY_LOG=<target>=<level>,…` | Log filter, e.g. `VICTORY_LOG=victory_loader=trace,info`; falls back to `RUST_LOG`. A release build cannot log below info whatever the filter asks |
| `VICTORY_STACKS=<s>,<s>` | Every thread's unwound stack as `module+rva`, that many seconds after start, to see what a title waits on |
| `VICTORY_WATCH=<rva>,…` | Polls the u32 at each title address (hex) every 5 ms and logs it whenever it changes — an engine's state, say |
| `VICTORY_CRASH_SAVE=<rva>:<len>,…` | Writes those title ranges (hex) to `crash-<rva>.bin` in the cache folder on a crash |
| `VICTORY_KEYS=<second>:<key>,…` | *diagnostics.* Presses each key for 0.2 s at that second after input starts, focus or not. A key is a letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`, `Left`, `Right`. This is how the account picker gets answered without a person: cover the whole window, `VICTORY_KEYS=5:Enter,10:Enter,15:Enter,…` out to five minutes, because the picker can ask twice |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | Records what the title plays, from our process only, as 48 kHz 16-bit stereo. It starts a child of the title to do the recording |
| `VICTORY_XMA_DUMP=<dir>` | Saves the first 32 XMA voices' context and input |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder.

### What a frame costs

These change what runs, not just what is logged. All are off by default unless
the row says otherwise, and the base's README is where their measurements live
(RTX 4070, in the world).

**Before reaching for any of them: on the adopted base a frame's cost is one
lock.** In a world-phase frame that logged **7,379.8 ms**, textures were
6,950.0 ms of it and **6,896.6 ms of those — 99.2 % — was a single wait at
`crates/victory-gpu/src/texture.rs:676`**, `waited(2, watch::title_memory_reading)`:
waiting for the runtime to stop remapping title memory, not copying anything.
Per byte that is 46 ms/MB against 0.59 ms/MB for a plain copy, eighty times, and
the copies themselves were 76.9 ms — about 1 % of the frame. So a log line whose
`textures` field is large does not say copying is the problem, and a change that
halves copy cost will not move a frame that is waiting. Taking that read once per
submission instead of once per texture is the fix under test.

| Variable | Effect |
|---|---|
| `VICTORY_PIPELINE_WAIT_MS=<n>` | The per-frame budget, over all threads, for waiting on a background pipeline. Default 500 ms, shared across the threads of a frame; a draw is omitted until its pipeline is ready, and a budget of 0 never waits |
| `VICTORY_PIPELINE_WAIT_BY_DURATION=1` | *diagnostics.* Off by default. On, a frame also gives up on a wait whose making is known to take longer than what is left of the budget — and **the draw it gives up on is the picture**: this is what turned a whole cutscene black on 2026-11-24. The measurement behind it stands (every world frame spent its 500 ms on stages that take 2.7 s and completed no pipeline), so it measures that, and it is never a setting |
| `VICTORY_PIPELINE_TIMING=1` | *diagnostics.* Prints, once a second, where a pipeline's time goes — queued, translated, waiting on another thread's translation, root signature, library load, driver compile, stored back — with the rate they add up to, what draining the queue would take at that rate, and how much of the time the workers were busy rather than waiting for a processor |
| `VICTORY_LATE_WAIT=1\|2\|3` | Sends a queue's submissions without waiting for each, and settles them at the next point a title buffer, a fence or a present is observed. Off unless the value is exactly 1, 2 or 3, and forced off by `VICTORY_GPU_TIMING`, which has to wait for each list to time it |
| `VICTORY_RELAY=<1\|2>` | A thread of its own lands finished flights as the card runs them, so the submitting thread never waits and copies back itself. `2` also waits only for the flights that declared a range, not for everything in the air |
| `VICTORY_WIDE_BACK=1` | Takes a submission's shader-written snapshots together and compares everything declared after the card, cut into ~1 MiB pieces over a few threads, instead of range by range on one |
| `VICTORY_CARD_BUFFERS=0` | Turns off card-side copies of large title buffers, which are on by default. `VICTORY_CARD_BUFFERS_MIN_MB` (16) is the smallest buffer that gets one and `VICTORY_CARD_BUFFERS_MB` (512) the most bytes spent on them |
| `VICTORY_INPLACE=<1\|2>` | Opens the title's own memory as a graphics heap and places buffers over it, instead of copying. `1` covers private windows outside buffers, `2` title buffers too — and `2` must not be combined with late wait, since a flight's rules know shader writes only by declared ranges |
| `VICTORY_BVH_BUILDER=1` | Lets the ray-tracing builder's stages run, which they do only while every builder pipeline the title created has a host pipeline. `VICTORY_BVH_QUARANTINE=1` also keeps the builder's stores out of title memory, for a first test. Nothing is lit by ray tracing yet |

One session at a time is a rule of the work, not a switch: the title shares one
GPU, one save folder and one shader cache, so two sessions at once is the one
failure this project cannot recover from — it blacked the reference machine
once. `translate`, `inspect`, `census`, `shader-info` and `lib-exports` are
offline and take nothing.

## Licence

**GPL-3.0-or-later.** See [LICENSE](LICENSE).

Everyone in the chain — upstream VIctory, fxceless/victory-gta6, and this fork —
licenses under the same terms, and the merged history keeps each author's work
identifiable by commit.

Redistributing this, in source or as a binary, means: ship the complete
corresponding source of the whole combined work under GPL-3.0-or-later, keep the
licence text and the copyright notices, and add no further restriction on top of
it. There is no warranty, and no game content is licensed here — the game stays
the user's own copy, obtained by them, and none of it is distributed with this
repository.
