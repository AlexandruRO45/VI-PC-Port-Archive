# Ray tracing in this fork: what the title does and what has been built

Status, 2026-11-25: **nothing is lit by ray tracing yet.** Most of the
machinery translates and is unit-tested; it has not yet run end to end in the
game. Everything here is off by default.

## What the title does

The title's indirect light (probes, bounce, reflections) and the
shadows of its lamps and sun are ray traced. Without them, anything the sun
does not hit directly is black: interiors, the shadowed side of a face, rooms
lit only by lamps.

The title does not use a ray-tracing API:

- Traversal is ordinary compute-shader code: a per-thread stack in shared
  memory, and loads through raw 40-bit addresses of title memory.
- The static world's acceleration structures ship pre-built in the title's
  data, in the console GPU's native node format (64-byte nodes: box nodes with
  four children and 16-bit float bounds, 128-byte box nodes with 32-bit float
  bounds, triangle nodes, instance nodes). They are unpacked on the GPU.
- A set of builder programs (from the title's `XboxBvhBuilder.dll`, 52 compute
  programs) builds and refits the dynamic structures on the GPU every frame.
- Only the per-node test is hardware: the instructions
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. One call tests a
  ray against a box node's four children (returning the hit children, nearest
  first) or against a triangle node (returning the intersection's numerator,
  denominator and barycentric numerators).

So mapping this onto the host's own ray tracing (DXR) is not practical: the
layer would have to know the geometry behind structures the title never
describes to it, and the traversal loops live in 64 of the title's own
programs. This fork runs the title's builder and traversal as translated
shaders and emulates the one instruction.

## What this fork adds

1. **The builder programs translate** (49 of 52; all that the title creates in
   the scenes tested). They pass buffer addresses in a spilled parameter table
   rather than in descriptors. Added: the spilled user-data table, root
   unordered-access views (translator and host binding), byte and 16-bit buffer
   accesses, and several control-flow shapes.
2. **Title memory by raw address from shaders.** A directory with one entry per
   64 KiB of title address space leads a shader to the layer's host copy of
   that memory; loads, stores, atomic add/sub and float min/max go through it.
   Before a dispatch the host makes ready the memory its root views and root
   constants name.
3. **The ray test, emulated in DXIL.** Four slab tests and a sorting network
   for box nodes; a watertight ray/triangle test for triangle nodes; both forms
   of the instruction. A plain Rust reference lives in
   `crates/victory-shader/src/bvh.rs`; tests run the emitted DXIL on the GPU
   over 311 ray/node cases and compare with the reference bit for bit.
4. **The trace and shading programs translate** (59 of 64 trace programs, and
   the irradiance-cache shading program). This needed lane masks treated as
   plain numbers on some paths, and exact forms for programs written for
   64-lane waves running on a GPU with 32-lane waves.
5. **A safety rule on the host side.** The builder's stages run only while
   every builder pipeline the title has created has a host pipeline; otherwise
   none run (`VICTORY_BVH_BUILDER=1` asks for it; `VICTORY_BVH_QUARANTINE=1`
   also keeps the builder's stores out of title memory, for a first test). A
   partly working builder wrote garbage bounds into title memory and crashed
   the title.

## Runs with the builder on (2026-11-25, Radeon RX 6800 XT)

- **All 52 builder programs translate and get host pipelines** (`victory
  translate --pipelines` on `XboxBvhBuilder.dll`), and with
  `VICTORY_BVH_BUILDER=1` the builder runs ("all of its programs have host
  pipelines").
- **The title then crashes within seconds** of the first builder frame, on
  several threads at once, each following a pointer field of one of its own
  objects into memory that holds `+inf`, `-inf` or a pair such as
  `0xff8000007f800000`.
- **`VICTORY_BVH_QUARANTINE=1` leaked**: copy-backs are declared per list, so
  another shader's declaration in the same list carried the builder's stores
  back too. Fixed (`HostList::quarantine`: a list with a quarantined builder
  stage copies nothing back). With the fixed quarantine the title ran 74 s with
  the builder on and no such crash, then the card was lost (`0x887A0006`) with
  frames at 1.1 s of GPU time.
- **What the title finds there is structure nodes** (`VICTORY_CRASH_REGMEM=1`,
  memory around the registers at a crash): 128-byte box nodes, four child
  words `0xffffffff` and four boxes, real (`(-1.30, -2.02, -2.85)` to
  `(2.03, 2.85, ..)`) or empty (`+inf` x3, `-inf` x3), where the title keeps
  tables of its own inside the structures and reads them on the CPU.
- **They come from the rehydrate stages.** The title rehydrates the
  dehydrated bottom-level structures it ships, in place, through the builder
  DLL's rehydrate entry point. That call writes nothing itself (every byte of
  the structure and of the 64 KiB after it is unchanged when it returns): it
  records builder stages into the command list. Each call names the
  structure twice, by its GPU address and by that address minus
  `0x6_0000_0000`, two views of the same pages (both mapped views here, with
  the same bytes), which is where the title reads it from on the CPU. The
  crash's object lies inside such a structure.
- **Dropping only those stages removes the crash**: with
  `VICTORY_BVH_SKIP_REHYDRATE=1` the builder's other stages run, nothing is
  quarantined, and the title ran on with no fault. The rehydrate records four
  builder programs (`28918b7150a7aad5`, `4ddd6f9702fab2a7`, `5e68767498c57eb6`,
  `caa78b8ca62bc3d6`).
- **One of them loses its reads**: with fault records naming the program and
  instruction, the lost raw-address accesses of a frame were nearly all by
  `5e68767498c57eb6`, at a dozen instructions, through real title addresses
  whose pages had no copy yet. Those reads give zeros, and a rehydrate runs
  once per structure, so the expanded nodes land where the zeros send them,
  over the tables the title then reads. The likely fix is to have every page
  a rehydrate stage reads ready before it runs (the extent of the title's
  allocation around each address, see the shortcut in `memory.rs`).
- **Ruled out on the way**: the first frames alone (`VICTORY_BVH_WARMUP=10`,
  quarantined for the builder's first 10 frames, crashes as soon as the
  results are let through); the float atomics (the compare-exchange loop the
  translator emits for the builder's 258 `global_atomic_fmin/fmax`, run by
  65,536 threads on one address in a custom write-back L0 buffer, as the 4 MiB
  windows are made, and in a default buffer, gives exact results).
  An earlier reading of `VICTORY_BVH_TRACE` blamed the title's own code: the
  infinities land at the structures' GPU addresses, and the crash reads them
  through the other view.

## Known gaps

- With the builder on, the rehydrate stages overwrite tables the title reads
  (above), until the pages they read are ready before they run.
- Five shadow-mask trace programs do not translate yet.
- An access to a title page that has no host copy is lost (a read gives zero, a
  store is dropped).
- Not publicly documented, and assumed: the instruction's rounding, the order
  of two boxes entered at the same distance, and the result for a ray exactly
  through an edge.
- Cost on the GPU is unmeasured.

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
- `crates/victory-loader/src/bvhcalls.rs` and `victory_core::bvh`
  (diagnostics, `VICTORY_BVH_SKIP_REHYDRATE=1`): rehydrate calls marked, the
  stages they record dropped (`pipeline::rehydrate_skipped`).
- `crates/victory-gpu/src/bvhtrace.rs` (diagnostics, `VICTORY_BVH_TRACE=1`):
  where infinities land in title memory, the arguments of builder dispatches,
  the copies reachable by address, and each lost access with its program
  (`victory_shader::emit::program_id`) and instruction offset.
