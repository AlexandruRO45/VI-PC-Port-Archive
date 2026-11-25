# Ray tracing

The local RTX 4090 prologue test on revision `05a78e3` ran with the ray-tracing
structure builder disabled. The captures showed stepped shadow edges and
missing or uneven lighting. See the [local test results](../README.md#local-test-results)
for the machine, configuration, and frame timing.

## What the title does

The title's indirect light (probes, bounce, reflections) and the
shadows of its lamps and sun use ray tracing. These effects contribute to
interior lighting, the shaded side of characters, and exterior shadows.

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

## Implementation

1. **Builder shader translation.** The shaders pass buffer addresses in a
   spilled parameter table
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
   `crates/victory-shader/src/bvh.rs` for comparison with emitted DXIL.
4. **Trace and shading shader translation.** The translator treats lane masks
   as scalar values where required and has dedicated handling for console
   shaders that use 64-lane waves on a GPU with 32-lane waves.
5. **A safety rule on the host side.** The builder's stages run only while
   every builder pipeline the title has created has a host pipeline; otherwise
   none run (`VICTORY_BVH_BUILDER=1` asks for it; `VICTORY_BVH_QUARANTINE=1`
   also keeps the builder's stores out of title memory for diagnostics).

## Known gaps

- With `VICTORY_BVH_BUILDER=1`, the host requires a translated pipeline for
  every builder stage the title creates before enabling its dispatches.
- An access to a title page that has no host copy is lost (a read gives zero, a
  store is dropped).
- Not publicly documented, and assumed: the instruction's rounding, the order
  of two boxes entered at the same distance, and the result for a ray exactly
  through an edge.
- The local frame timings were recorded with the builder disabled.

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
