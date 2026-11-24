# Working together on this checkout

More than one agent works this tree at the same time. This file is how they
avoid colliding; it is not a build or contribution guide (README.md and
docs/ have those). Keep it short, timestamped and current.

## One title session at a time

The title shares one GPU, one save folder and one shader cache, so only one
agent may run `victory.exe run` at a time.

- Before launching, check `D:\victory-session.lock`. If it exists and is
  younger than 30 minutes, someone else is running: wait or work offline.
- Write it as `<agent> <what> <started, ISO-8601>` while your session runs,
  and delete it when the process exits.
- A lock older than 30 minutes is stale: take it over.

Offline work never needs the usual lock: `victory.exe translate <paths>`
(including on `VIctory-data\failed-shaders`, where a diagnostics run dumps
every container that failed), `cargo build`, `cargo test`.

## Claims (who is editing what, right now)

| Area | Owner | Notes |
|------|-------|-------|
| `victory-shader` descriptor/stride handling, address analysis, mask spills, raw heap views | Codex agent (main checkout `VIctory`) | Its uncommitted work in `lower.rs`, `resource.rs`, `emit/mod.rs`, `victory-gpu/src/descriptor*` |
| Host GPU rendering, black-frame captures and pipeline warm-up diagnosis (started 2026-11-24T17:20+01:00) | Codex agent (main checkout `VIctory`) | No title session active; captures in `VIctory-data/render-capture`; acquire `D:\victory-session.lock` before the next `victory.exe run` |
| Clamped image samples (`image_sample*_cl`), byte/half buffer loads (`buffer_load_ushort` and friends), biased 16-bit samples, pixel barycentrics (`SV_Barycentrics`) | Particle agent (worktree `D:\victory-gfx`, patch `D:\victory-emit-gaps.patch`, 218 lines, 343 tests green) | Merged into the main checkout (patch needed one follow-up in `shader.rs`: the `barycentrics` field and `flags::BARYCENTRICS`); byte/half *stores* added on top |
| Compute translation: descriptor-from-computed-index, root-argument pointers, structured element views, `ds_bpermute_b32`, `docs/compute-gaps.md` | DGR-3 / Particle agent (worktree `D:ictory-wt-compute-descriptors`, `cap/compute-descriptors`) | Main landed the host half of the pointer register first (`e3dac6e`), so this lane adds no `victory-gpu` change |
| Cube samples with explicit gradients (`pixel-2deec721a63fe8a5`, 3 programs) | Particle agent (lane B, `D:ictory-wt-emit-gaps`) | Closed in `c2b1843` (`cube_gradient`): the four derivatives are the face-coordinate (ds,dt) pair of each axis, and the emitter applies the per-face Jacobian of its own direction table |
| Lane spill of a scalar read back at a constant lane (pixel-ee41ebc3eb3fbbdc, the last pixel refusal in the dump sets) | Particle agent (lane B, worktree `D:ictory-wt-emit-gaps`) | `v_writelane_b32` then `v_readlane_b32` at a constant lane is that scalar, in a pixel program too (no wave read); reuses `spilled_value` |
| Mesh wave size (all 8,383 mesh programs) | Particle agent (lane B, worktree `D:ictory-wt-emit-gaps`) | DXIL allows the wave-size attribute only on compute and node shaders, so every mesh program emitted a module the validator rejects; none is emitted now, `PSV0` says any wave, and host mesh pipelines still create |
| Lane masks read as data in mesh (207 programs) | Particle agent (lane B) | Refused, with the reason and what would settle it in `docs/emit-gaps.md`: a mesh stage cannot declare its wave, so its lanes are the device's |
| Lane masks read as data (`s_xor_b32 m, -1` complements), nested descriptors as scalar SRDs | Particle agent (worktree `D:ictory-gfx`) | Six compute programs no longer stop on the mask: the complement's number is the base mask's ballot over the whole wave. They now stop on a descriptor's address or format read as data, which is the resource lane |

Update your row when you start or finish something, and add a row rather than
editing someone else's.

## Known blockers for the world rendering

Measured on the `failed-shaders` dump set, main checkout:

- the world's main pixel shader `pixel-a7a20f06a33260f9` emits now that the
  descriptor work landed;
- `pixel-2deec721a63fe8a5` is the last pixel gap: cube sampling with
  explicit gradients, where the console passes five of six derivative
  components;
- the compute side is address analysis plus descriptor-from-a-computed-index,
  and those computes are what fill the indirect draw arguments: with them
  dropped, the world's draws run with empty records and the scene target
  stays black (measured: the 3840x2160 D32 depth target is untouched while
  the 8-bit UI target holds content).

Scratch files that the policy test refuses live in `D:ictory-scratch`
(run logs, PR patches, `.rej` and `.codex-backup` leftovers), not in here.

A black frame in a diagnostics run is a frame whose draws were dropped for
want of a host pipeline: `host: dropped N/frame without a host pipeline` in
the log names the shader hashes.

## Findings from the emitter side (Particle, offline, 2026-11-24)

- Descriptor-as-SRD: a program that loads another load's descriptor dwords
  out of memory and uses them as a scalar buffer's descriptor no longer
  stops translation (lower.rs `descriptor_data`: the `s_buffer_load_dword*`
  consumer's descriptor is a host handle, so nothing is read as a number).
  `compute-3fa1fda35402c93f` now reaches its real blocker, a descriptor's
  address read as data at 0x3628.
- The two unimplemented device slots are not the crash. Both call sites test
  the HRESULT and fall back to a null interface: at
  `0x1424800d5` the code is `mov rax,[rcx]; call *0x218(rax); mov esi,eax;
  test esi,esi; js -> xor eax,eax`, with `r8 = 0x1461e06b8` (a 16-byte id,
  `765a30f3-f624-4c6f-a828-ace948622445`, not a D3D12 device IID) and
  `r9 = &out`. So the trap at `+0x23cf683` asserts on state the title got
  from somewhere else; chasing these slots is wasted work.
- Cube sample with gradients: the console's address list is three direction
  components, four derivative components (two per axis, face space) and one
  clamp dword for the `_cl` form, which is exactly the five-of-six seen.
  Reconstructing direction-space gradients needs a per-face Jacobian; a
  guess only moves mip selection, so it needs a look at the title.
- Mesh path, measured from the run logs: host mesh pipelines are created
  (2212 in one run) and mesh dispatches run (300-400 per run), so mesh
  shaders, mesh pipelines and `DispatchMesh` all work end to end, and the
  pipeline library hits (loaded 541 against 291 made over a run). The dump
  set's five mesh programs fail in the resource layer: four read a descriptor
  from per-lane computed VGPRs (a waterfall) and one joins a descriptor
  partly by control flow. That, not the mesh pipeline, is what stands
  between the title's mesh work and the screen.
