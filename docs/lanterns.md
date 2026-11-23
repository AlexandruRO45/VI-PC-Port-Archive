# Lanterns: a synthetic title through `gta6rt --run`

Lanterns is a test guest, an x64 Windows program shaped like a title. It goes through the same chain a title would:

1. Its RDNA2 shaders are hand-written. They are packed into synthetic FSSM programs inside the container the title
   ships: `DXBC { DXIL { XDAT { 0x0D: XBCH + zlib(FSSM) } } }`.
2. The project's offline tools translate them, the way `make_game.py`'s shaders stage does: `fssm_unpack.py`,
   `shader_disasm.py` (code range from the FSSM header) and `rdna2spirv.py`.
3. `gta6rt --run` maps the guest and answers its imports. The guest gets its device only from
   `d3d12_xs.dll!D3D12XboxCreateDevice`, which the shim registry answers with the Vulkan device
   (`LEONIDA_D3D12=vulkan`).
4. The guest creates its pipelines from those containers through the D3D12 API. The layer finds each translated shader
   by the SHA-256 of its FSSM program.
5. The guest draws, reads the frame back and compares every pixel with a CPU reference. A negative control then shows
   that the check would have caught a wrong frame.

Nothing comes from the game. The scene is abstract solids on a checkered floor, and the shaders, scene and reference
are ours.

Code: `runtime/test/lanterns/`. `lanterns.cpp` is the guest and `lanterns_gpu.cpp` its D3D12 side.
`lanterns_scene.h` is the scene and `lanterns_reference.h` the CPU reference. `shaders/*.asm` are the shaders,
`fssm_pack.py` the packer, `run_lanterns.py` the runner and `lanterns.cmake` the build.

## How it relates to the other synthetic tests

Each boot milestone in [d3d12_vulkan.md](d3d12_vulkan.md) section 6 has its own gate: `d3d12_draw_test` (M1),
`d3d12_translated_draw_test` (M1.5), `d3d12_m2_textured_test`, `d3d12_m3_frame_test` and `d3d12_m4_compute_test`. These
are host programs. They link the layer and create the device directly, and most build their shaders with
`runtime/test/shaders/build.py`, which hands llvm-mc's listing straight to the translator.
`run_tiny_frame` (#3488) runs a mapped guest through `gta6rt --run` for a clear and a readback.

Lanterns puts the pieces together in one guest. Its shaders take the title's path through the FSSM container,
`fssm_unpack.py` and `shader_disasm.py`, and the guest runs under `gta6rt --run`. Each stage then adds what the
milestone tests check one at a time.

## Stages

Each stage is its own run (`--stage N`) and prints its own PASS line. Every stage has a control the reference
predicts, so a check that could not tell a right frame from a wrong one fails.

| Stage | What it adds | Control |
|---|---|---|
| [1](#stage-1) | Five indexed, instanced draws with a depth buffer | Depth test off changes the predicted pixels |
| [3](#stage-3-compute-on-a-second-queue) | A compute queue moves the points of light; the frame waits on its fence | Point states and records equal the CPU's; the points move the predicted pixels |
| [5](#stage-5-an-fp16-target-and-a-bloom-post-process) | An fp16 target, then a bloom whose waves share LDS across an `s_barrier` | Skipping the blur changes the predicted pixels |
| [6](#stage-6-compressed-textures-with-mip-chains) | BC1 and BC7 textures with full mip chains, trilinear and explicit-LOD sampling | Two mips down leaves the predicted pixels' bounds |
| [8](#stage-8-a-shadow-map) | A D32 shadow map with PSO depth bias, sampled with 2x2 PCF | The map cleared to far changes the predicted pixels |
| [9](#stage-9-deferred-lighting-with-tiled-light-culling) | A G-buffer of three render targets, then 256 point lights culled per 16x16 tile in LDS | One light moved changes the predicted pixels; culling off gives the same image |
| [10](#stage-10-4x-msaa-resolved-with-resolvesubresource) | The frame drawn at 4x MSAA and resolved with `ResolveSubresource` | The 1x render differs on exactly the predicted edge pixels; `SampleMask` isolates each sample |
| [11](#stage-11-ray-traced-shadows-through-blastlas) | One shadow ray per pixel through a TLAS over two BLASes, walked with `image_bvh_intersect_ray` | Masking one instance or swapping two BLAS references changes the predicted pixels; a stalled BVH stops at the step cap |
| [20](#stage-20-samplers-held-as-literals) | A compute shader samples through three samplers it builds from immediates, as the console compiles static samplers | A POINT sampler made LINEAR, or the comparison bits cleared, changes the predicted samples |
| [22](#stage-22-a-g-buffer-round-trip) | Five and six G-buffer targets written through packed half exports, then decoded by a compute pass | A binary16 target in place of the RGBA8 IDs, or UNORM in place of sRGB, breaks the predicted values |
| [23](#stage-23-alpha-tested-ray-traced-shadows) | Stage 11's shadow rays through non-opaque BLASes, a hit counting only on the opaque side of its barycentrics | u and v swapped in the shader moves the predicted shadows |
| [24](#stage-24-local-x-and-y-thread-ids) | A compute shader with a 16x8 local size reads local X and local Y | X and Y swapped moves every predicted word |

## Stage 1

Five `DrawIndexedInstanced` calls draw hexagonal prisms, cubes, octahedra, pyramids and a checker floor into a 384x256
`R32G32B32A32_FLOAT` target with a `D32_FLOAT` depth buffer (depth test LESS). The frame is drawn twice: once with the
depth test (the image) and once without it (the negative control).

What it covers:

- The PE loader, the shims and `D3D12XboxCreateDevice` on Vulkan, called from a mapped guest.
- Pipelines from DXBC/XDAT/XBCH containers, with each shader found by its FSSM hash. The VS container has two 0x0D
  payloads, as 195 of the title's do, and the layer must take the last one, as the tools do.
- Wave32 and wave64 shaders. The octahedra are drawn with a wave64 VS and PS pair; every VS in the title's census is
  wave64.
- The title's register layout. The title's VS and PS read their first table pointer at `s8`
  ([d3d12_vulkan.md](d3d12_vulkan.md) 3.5). The layer packs root arguments from `s0` in root-signature order, so a
  root signature that starts with eight 32-bit constants puts them in `s0`..`s7` and the first table at `s8`. A
  second, small draw does exactly that: a wave64 VS and PS (`vs_lanterns_edges`, `ps_lanterns_edges`) read their
  constants from `s0`..`s3` and their table from `s[8:9]`, the VS its vertices through slot 0 and the PS one value
  through slot 1.
- Vertex, index and instance buffers in DEFAULT heaps, filled from an upload buffer by `CopyBufferRegion`.
- The vertex shader fetches the vertex and the instance through two V#s in two descriptor tables. It reads the descriptor
  heap start handles through heap slots 9 and 10.
- The model matrix and tint come from 16 root constants, and the camera from a root CBV.
- The floor's pixel shader reads its two checker colours from its own FSSM program, through the code base
  (`s_getpc_b64`).
- The PS input table declares one flat input and one interpolated input. On the floor shader their slot bytes are
  6 and 3, not position + 1: the attr number is the table position, so a reader that used the slot would fail.
- Clear, depth, readback through `CopyTextureRegion`, and a fence wait.

How the check works:

- **Exact pixels.** Colours are short binary fractions, so every product the shaders form is exact in float. Pixels
  that are clearly inside one face, or clearly background, must match the CPU reference bit for bit.
- **Uncertain pixels.** A few pixels depend on how the GPU rounds: triangle edges, depth near-ties and checker borders.
  Each of these must be one of the colours the reference lists for it. The background is on that list only where no
  surface certainly covers the pixel. Near an edge that two faces of one solid share, one of the two must cover it,
  because both get the same vertex positions, so a crack there fails.
- **Shared edges.** The second draw is a fan of eight triangles around one vertex, every vertex on a pixel centre, with
  inner edges horizontal, vertical and at slopes 1:1, 1:2, 3:2 and 1:4. 673 pixel centres lie exactly on an edge two
  triangles share. Each triangle has its own flat colour, and additive blending counts how often each pixel is drawn.
  The CPU applies D3D's top-left rule in exact integers (`lanterns_edges.h`). Every one of the 98,304 pixels must
  match: each shared-edge centre drawn once, by the triangle the rule names, no pixel drawn twice, no gap.
- **Lookup control.** Before drawing, the guest writes two containers of the vertex shader's FSSM program with the
  same writer (`runtime/test/d3d12_test_shaders.h`). The first holds the program as it is, and the layer must give a
  pipeline for it. The second has one byte of the program's name changed (`vs_lanterns` to `vs_lanternS`), so its
  FSSM hash is one no translated shader has, and the layer must refuse it with `E_INVALIDARG`. The two containers
  differ only in that byte, so the refusal comes from the lookup, and a pass cannot come from a lookup that matches
  anything.
- **Negative control.** With the depth test off, submission order decides what is visible. The reference predicts
  which certain pixels change. The GPU frame must change exactly those pixels, and there must be at least 1,000 of
  them. A layer that ignored the depth state, or a reference that could not tell the two frames apart, fails here.

On success the guest prints one line and exits 0:

```
Lanterns stage 1: PASS, 97926 of 98304 pixels exact, 378 edge pixels plausible, depth-off control changed 11610 pixels; shared edges through a wave64 VS/PS with the title's s8 table: 98304 of 98304 pixels exact, 673 centres on shared edges each drawn once
```

`run_lanterns.py` passes the run only when all of these hold: exit code 0, this line exactly once, whole and with
numbers that fit together (exact and edge pixels add up to the frame, at least 1,000 changed), and the shim's
complete factory line, `[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy 0x... (Vulkan device 0x...)`. That last line
means the device is the Vulkan one, not the host's. The guest waits for each fence for at most 30 s and fails with a
line saying so. With `--wine`, the runner first stops the prefix's `wineserver` (`wineserver -k`, then `-w`,
10 s each, killed with everything it started if it overruns): a launch under a new virtual X server while the last run's `explorer.exe` is still alive hangs in
winevulkan's device creation, because that explorer keeps the old, closed display. The runner's timeout kills everything the run started, so a stalled guest, Wine or GPU fails the run
with exit code 3 instead of hanging it. On Linux it also catches a process that left the launcher's session (it makes itself a child subreaper, and if the kernel
refuses that it does not launch at all and exits with code 4); if
anything survives the kill, or taskkill fails on Windows, the run fails with exit code 4 and names what is left.
Each run writes into a new folder under `--out`: the guest's dumps, the launcher's output, `lanterns-stage1.png`
and `lanterns-stage1-sheet.png`. The sheet shows the GPU frame, the reference, the control frame, a mask (grey:
exact, red: wrong, yellow: uncertain), and the edge fan from the GPU and from the CPU (each triangle in its own
colour, white where drawn twice, black where not drawn).

Each check was made to fail on purpose, and each fix in it was shown to matter:

- A wrong shader: the floor shader repacked with one checker colour changed (red 0.4375 to 0.5). The guest reported
  20,430 certain pixels wrong and exited 1.
- A crack: every face of the cubes pulled 0.16% of the way to its centre, so a sliver of background shows along the
  edges, less than 1/32 pixel wide. The guest fails, on an edge pixel the reference gives to the pair of faces. With
  the background still a candidate there, as before, the same frame passed.
- A lookup that refuses the guest's container: the runtime patched to refuse a DXBC container whose size field is 0.
  The guest fails at the unmodified container's pipeline. With only the renamed container, as before, it passed.
- A fence nobody signals: frame 2 of stage 3 made to wait for one value more. The guest fails after 30 s with the
  fence line. Before, the run hung until the runner's timeout.
- The register layout and the fill rule: the edge PS reading table slot 0 instead of slot 1 leaves 31,884 of
  98,304 pixels exact. A reference with a bottom-right rule instead of top-left leaves 0 of 673 shared-edge centres
  right (319 gaps). Both fail.
- A stale `wineserver`: seen on another Linux setup, where 20 of 20 launches hung without the stop and 20 of 20 passed
  with it. On the test machine two back-to-back runs passed either way, so here it rests on the runner's test with a
  fake `wineserver` on `PATH`.

## Stage 3: compute on a second queue

A compute shader, `cs_lanterns_rise.asm` (wave32), runs on a COMPUTE queue. It moves 256 points of light in integer
fixed point, so the CPU can repeat every step exactly. It writes their instance records into a DEFAULT buffer with UAV
access. The direct queue waits on the compute fence and draws the points as a sixth `DrawIndexedInstanced`, whose
instance V# points at that buffer.

- Frame 1: the compute work (24 steps) is submitted, then the frame.
- Frame 2: the frame is submitted first. It waits for a fence value that a later compute submission (24 more steps)
  signals, so the layer must hold the direct queue until then.

What it adds to stage 1:

- A compute pipeline from an FSSM container. The thread-group size and `COMPUTE_PGM_RSRC2` come from the FSSM header.
- A compute root signature with four root constants and a table of two UAVs, as in `d3d12_m4_compute_test`.
- `Dispatch` on a compute queue, and `Wait`/`Signal` across the two queues.
- A draw whose instance data is what the compute shader wrote.

How the check works:

- The point states and the instance records are read back. They must equal the CPU's simulation of the same steps,
  word for word.
- Both frames are checked as in stage 1: certain pixels exact, uncertain pixels one of their candidates.
- Between frame 1 and frame 2 the points must change exactly the certain pixels the reference predicts.
- Control: a third frame without the `Wait`, submitted before its compute work. It is reported, not required, because
  a layer may order the queues by itself. On the test machine the result varies from run to run: the previous frame's state,
  or a mix of the two (16,183 and 16,546 certain pixels differed from the new state in two runs).

```
Lanterns stage 3: PASS, 97675 and 97664 of 98304 pixels exact, 256 points exact after 24 and 48 compute steps, points moved 14839 pixels
```

## Stage 5: an fp16 target and a bloom post-process

The stage-3 frame after 48 compute steps is drawn into an `R16G16B16A16_FLOAT` target. Then a post-process runs on
the same direct queue (`lanterns_post.h`):

- Two compute passes blur what is brighter than a threshold, first along rows and then along columns. Each group
  is 128 lanes sharing LDS across an `s_barrier`. The row pass reads the scene through a T# with `image_load`.
- The row pass is wave32 (four waves), and every sample a lane sums was loaded by a lane of another wave (lane i
  loads sample i + 64 mod 128; the halos come from the other half too). The second half also walks four dependent
  fetches before its write, so its write lands late. Only the `s_barrier` orders those writes before the reads.
- The column pass is wave64 (two waves), translated with `--wave64-emulate`. The emulation runs a wave64 program in
  rounds that already put every LDS write before a later LDS read, so there a missing `s_barrier` cannot show; that
  is why the row pass, the one whose barrier is measured, is wave32.
- A fullscreen pixel shader reads `SV_Position`, the scene texel and the blur, and writes their sum into an
  `R32G32B32A32_FLOAT` target.


How the check works:

- The fp16 scene is checked as in stage 1, against the reference rounded to half.
- The final frame must be exact where it is certain, and within bounds where the scene's edges leave it uncertain.
- The point states and instance records must equal the CPU's, as in stage 3.
- Control: the composite with the blur skipped must change exactly the certain pixels the reference says the bloom
  changes, and at least 1,000 of them.
- Measured, not required: the row pass without its `s_barrier` (a race the reference cannot predict). On the test
  machine it changed 8,449 to 14,849 certain pixels in five runs; the same shader without the barrier in place of
  the real one fails the stage.

```
Lanterns stage 5: PASS, fp16 scene 97664 of 97664 certain pixels exact, bloom composite 82563 of 82563 certain pixels exact and 15741 bounded pixels within bounds, skipping the blur changed 15113 pixels as predicted
```

## Stage 6: compressed textures with mip chains

The stage-1 scene, textured. `lanterns_bc.h` makes two patterns from integer formulas (a 256x256 mosaic and 64x64
nested squares), box-filters them down to 1x1 and compresses them with its own encoders: BC1, and BC7 mode 6. The
textures go up the D3D12 way: `GetCopyableFootprints`, an upload buffer, one `CopyTextureRegion` per mip level. No
tiled layout is written or assumed. The floor samples the BC1 mosaic with `image_sample` (trilinear, LOD from the 2x2
quad's derivatives). The solids sample the BC7 squares with `image_sample_l`, at a mip level each draw passes in a root
constant. The root signature gains an SRV table and a sampler table.

How the check works. Texture filtering precision differs between GPUs, so the reference decodes the same blocks and
gives every pixel bounds instead of one colour (`lanterns_sampling.h`):

- Solids: the exact bilinear sample, plus or minus 4/255.
- Floor: the trilinear sample over the whole LOD range Vulkan allows (coarse or fine derivatives, and a scale factor
  anywhere from the largest component to their sum), plus or minus 4/255.
- Control: the solids drawn two mip levels down must leave the main bounds wherever the two references cannot overlap.

```
Lanterns stage 6: PASS, 98144 of 98144 certain pixels within the BC1/BC7 sampling bounds (trilinear floor, image_sample_l solids), control two mips down moved 10221 of 10221 predicted pixels
```

## Stage 8: a shadow map

A depth-only pass draws the solids from a directional light into a 512x512 `R32_TYPELESS` map. It uses
`vs_lanterns`, no pixel shader and no render target, and the depth bias comes from the PSO (`DepthBias` 1000,
`SlopeScaledDepthBias` 3). The map has a `D32_FLOAT` depth view and an `R32_FLOAT` shader view. The main pass's
pixel shader, `ps_lanterns_shadow.asm`, moves the world position into light space and takes four comparison samples
half a texel apart (`image_sample_c`, `LESS_EQUAL`, point filtering: 2x2 PCF). It scales the face colour by
3/8 + 5/32 per lit sample.

How the check works. `lanterns_shadow.h` rasterizes the map on the CPU. It bounds each texel's stored depth
(coverage, tolerance, and any bias D3D12 and Vulkan allow) and each sample's texel. A pixel whose four samples are all
certain has one exact colour; every other certain pixel must be within its bounds. Control: the depth pass without
casters leaves the map cleared to 1, and that must change exactly the pixels the reference says are shadowed.

```
Lanterns stage 8: PASS, 95699 of 95699 single-colour pixels exact and 98144 of 98144 certain pixels within bounds (D32 shadow map, PSO depth bias, 2x2 PCF with comparison samples), the map cleared to far changed 9539 pixels as predicted
```

## Stage 9: deferred lighting with tiled light culling

The stage-1 scene drawn into a G-buffer of three render targets in one pass: `R8G8B8A8_UNORM` albedo, `R10G10B10A2_UNORM`
normal and `R32_FLOAT` view depth. `vs_lanterns_gbuf` exports each face's normal, packed into its vertices' shade word
as an exact integer float, and `ps_lanterns_gbuf` unpacks it with `v_bfe`. Then `cs_lanterns_tiled` runs one 256-lane
workgroup (eight wave32 waves) per 16x16 tile:

- each lane rebuilds its pixel's world position from the view depth and folds it into the tile's box in LDS with
  `ds_min_i32`/`ds_max_i32` on order-preserving float keys;
- lane i tests light i's sphere against the box, and a kept light takes a list slot with `ds_add_rtn_u32`;
- every lane loops over the list and adds each light's contribution in integers.

How the check works. `lanterns_deferred.h` holds the lights, the root constants and the CPU reference. Each light's
contribution is an interval, because the rebuilt position is known only within the depth's snapping slack.

- The albedo target must be exact on every certain pixel; probes of the normal codes and the view depth must be within
  their bounds (D3D12 lets a float-to-UNORM conversion be off by 0.6 ULP).
- Every single-colour pixel of the lit frame must be exact, every other certain pixel within its bounds.
- Each tile's list must be exactly the set of lights whose sphere meets its box. The one tolerance: a light within 2e-3
  of the box test's threshold (the GPU forms the same numbers in float) may go either way. The PASS line gives how many
  light-tile pairs are in that band (6 of 98,304 here), and more than 32 fails, so a culling error hiding inside the band
  would have to stay that small.
- The output and list buffers start poisoned, and with culling off (every light in every tile) the image must be the
  same bit for bit.
- Control: one light moved must change exactly the pixels the reference predicts and keep every other pixel.

```
Lanterns stage 9: PASS, G-buffer of 3 MRTs (RGBA8 albedo 98144 of 98144 exact, RGB10A2 normal and R32F depth 52643 of 52643 probes), 75462 of 75462 single-colour pixels exact through tiled culling with LDS atomics (10.4 of 256 lights per tile, the exact set in 384 of 384 tiles, 6 light-tile pairs within 2e-3 of the box test left free), culling off gave the same image bit for bit from poisoned buffers, one light moved changed 1231 of 1231 predicted pixels
```

## Stage 10: 4x MSAA resolved with ResolveSubresource

The stage-1 scene drawn into a 4x `R32G32B32A32_FLOAT` target (`SampleDesc.Count = 4` on the target, the depth buffer
and the pipelines) and resolved into the single-sample target with `ResolveSubresource`.

How the check works. `lanterns_msaa.h` rasterizes the reference once per sample at D3D12's standard 4x positions
(coverage and depth at the sample, shading at the pixel centre, as a pixel shader run once per pixel interpolates) and
averages the four. A pixel whose samples are all certain has one expected colour; for the others the reference
enumerates every average their sample candidates can give.

- Every certain pixel must be exact, including the edge pixels whose samples differ and blend.
- Every uncertain pixel must be one of its enumerated averages.
- Control: the same frame at 1x must differ from the resolve on exactly the predicted edge pixels and nowhere else
  certain.
- `SampleMask` set to `1 << s` in four more pipelines: each frame must match sample s's own reference, including
  pixels that only that sample's position explains.

```
Lanterns stage 10: PASS, 4x MSAA resolved by ResolveSubresource: 97438 of 97438 certain pixels exact at the standard sample positions (1173 edge pixels blended exactly), the 1x render differs on 1068 of 1068 predicted edge pixels and nowhere else certain; 866 of 866 uncertain pixels are one of their enumerated averages; SampleMask 1 << s isolated each sample (at least 188 pixels only that sample's position explains)
```

## Stage 11: ray-traced shadows through BLAS/TLAS

`BuildRaytracingAccelerationStructure` builds one bottom-level structure per traced mesh (hexagonal prisms and square
columns) and top-level structures over their ten instances. `cs_lanterns_rt_shadow` then casts one shadow ray per pixel
of a stage-9 G-buffer towards the light with `image_bvh_intersect_ray`:

- each lane keeps its own stack in LDS, and every push checks it; a lane that would overflow stops and is marked;
- TLAS boxes are tested with the world ray; instance records are read through a raw V# over the TLAS (mask, BLAS
  pointer, world-to-object), and the ray moves to object space for that instance's BLAS;
- the loop stops at a step cap (2048); a ray still tracing or overflowed is drawn magenta, which no reference allows.

How the check works. `lanterns_rt.h` holds the instance transforms and a CPU tracer (Moller-Trumbore in double with a
barycentric margin), run from the corners of each pixel's position box (the depth's snapping slack).

- Every single-colour pixel must be exact, every other certain pixel within its bounds.
- Controls: a TLAS with one instance's mask at 0, and one where two instances swap BLAS references, must each change
  exactly the predicted pixels and keep every other.
- A BVH whose root box is its own only child must stop every traced ray at the cap, and every background lane must
  report 0 steps; a root box that is its own four children must stop every ray as overflowed.
- The output and step buffers start poisoned; no poisoned word may remain.

```
Lanterns stage 11: PASS, one shadow ray per pixel through a TLAS of 10 instances over 2 BLASes built by BuildRaytracingAccelerationStructure, traversed with image_bvh_intersect_ray: 97452 of 97452 single-colour pixels exact, masking one instance changed 1058 of 1058 predicted pixels, swapping two BLAS references changed 966 of 966; the step cap stopped 54348 of 54348 stalled rays and a full stack 54348 of 54348
```

## Stage 20: samplers held as literals

The console's compiler turns a root signature's static samplers into literal S#s: four `s_mov_b32` immediates, then
`image_sample`. Nothing in a heap or a root argument names them, so the layer has to register the words it finds in the
shader when the pipeline is created. `cs_lanterns_samplers` holds three, all clamping, with `MIP_FILTER` point and
`MIP_POINT_PRECLAMP` set as the title's literals have them: POINT, LINEAR (bilinear) and a LESS_EQUAL comparison with
point filtering. The root signature also declares them as static samplers, as a title's does.

One wave32 group, one probe per lane, explicit LOD 0 (`image_sample_l`, `image_sample_c_l`):

- a 4x4 `R32_FLOAT` colour texture (texel (x, y) holds 1 + x + 4y) through POINT and through LINEAR;
- a 4x4 depth texture (`R32_TYPELESS`, cleared quadrant by quadrant to 0.2, 0.4, 0.6 and 0.8 with rectangle clears, read
  through an `R32_FLOAT` view) through the comparison sampler, and through POINT.

How the check works. `lanterns_samplers.h` computes every expected value. Each probe sits a quarter or three quarters
of a texel from the texel edges, so POINT meets no tie and the bilinear weights are 1/4 and 3/4; with small-integer texels
every expected value is exact.

- Every sample must have the expected float's bits.
- LINEAR must differ from POINT at the probes where the reference says it does (30 of 32).
- The comparison must give 0 or 1 (both occur), and a plain POINT sample of the same depth texture must give the depth:
  a plain sample must not become a comparison by accident.
- The output buffer starts poisoned.

Mutations that fail: the POINT literal made LINEAR (`0x14000000` to `0x14500000` in its third word) gives 2 of 32 POINT
probes exact; the comparison bits cleared (`0x3092` to `0x92`) gives the raw depths, 0 of 32 comparisons exact.

```
Lanterns stage 20: PASS, three samplers the shader holds as literal S#s (as the console's compiler writes static samplers): POINT 32 of 32 probes exact, LINEAR 32 of 32 (differing from POINT at 30 as predicted), LESS_EQUAL comparison 32 of 32 (12 pass, 20 fail), and a plain POINT sample of the same depth texture 32 of 32 raw depths
```

## Stage 22: a G-buffer round trip

The title's deferred pixel shaders write five or six targets and pack every colour export as half floats
(`v_cvt_pkrtz_f16_f32`, `exp ... compr`); a lighting pass reads them back. `ps_lanterns_gbuf6` (fullscreen, with
`vs_lanterns_full`) writes six targets from the pixel's integer position:

| Target | Format | Holds |
|---|---|---|
| 0 | `R8G8B8A8_UNORM` | four byte IDs |
| 1 | `R10G10B10A2_UNORM` | three 10-bit codes and a 2-bit alpha |
| 2 | `R16G16_FLOAT` | signed motion |
| 3 | `R8G8B8A8_UNORM_SRGB` | albedo, written linear and stored sRGB-encoded |
| 4 | `R16G16B16A16_FLOAT` | emissive |
| 5 | `R8G8B8A8_UNORM` | four more byte IDs |

It runs twice: with all six targets bound, and with a 5-target pipeline, where the sixth export goes nowhere.
`cs_lanterns_gbuf_decode` then reads targets 0, 1, 2, 4 and 5 through T#s and decodes them as a lighting pass does: byte
IDs by multiplying by 255 and truncating, codes by rounding, and a lighting term from the decoded values.

How the check works. `lanterns_gbuf6.h` computes every target's bytes and the decode's output, from the f16 values the
exports round toward zero. Gated separately, over both runs:

- byte IDs (targets 0 and 5, and their decode), the 10-bit codes, the motion and the emissive values: exact;
- the sRGB albedo: within one code of the CPU's encoding, as D3D allows (the PASS line says how many are exact);
- the decoded lighting: exact;
- with 5 targets, the sixth target stays at its clear value.

The byte IDs survive only because their target is RGBA8: an f16 rounded toward zero and decoded by multiplying by 255 and
truncating loses one in 254 of 256 values. Mutations that fail: target 0 made `R16G16B16A16_FLOAT` (0 of 196,608 byte
IDs right, the lighting at 1,536); the albedo target made UNORM instead of sRGB (22 of 196,608 albedo pixels within one
code). The sRGB part follows #4251, which checks the sRGB formats the game uses.

```
Lanterns stage 22: PASS, G-buffers of 5 and 6 targets written through packed half-float exports: byte IDs 196608 of 196608 pixels exact (RGBA8, read back by a truncating x255 decode), 10-bit codes 196608 of 196608, signed RG16F motion 196608 of 196608, RGBA16F emissive 196608 of 196608, sRGB albedo 196608 of 196608 within one code (178702 exact), decoded lighting 196608 of 196608 exact; with 5 targets the sixth stayed clear at 98304 of 98304 pixels
```

## Stage 23: alpha-tested ray-traced shadows

The title's foliage and fences are alpha-tested in its ray-traced shadows: the BLASes are built from non-opaque
geometry, and a hit counts only where the surface is opaque at the hit. `cs_lanterns_rt_alpha` is stage 11's shader
with that test. The barycentrics `image_bvh_intersect_ray` returns (u = i_num / t_denom, weighing the triangle's
second vertex, v = j_num / t_denom, its third, as D3D12 defines them) must give u + 2v < 1; the rest of every triangle
is cut out, and the traversal goes on past a hit there. The rule is not symmetric in u and v, so it also checks that the
two words come back in D3D12's order (the translator reads the node's vertices and applies the leaf's remap byte; see
[bvh.md](bvh.md)).

How the check works. `lanterns_rt.h` traces the same rays on the CPU and applies the same rule. A hit within 0.01 of
u + 2v = 1, or of a triangle edge, leaves its pixel undecided. Every single-colour pixel must be exact. The same frame
is also traced with stage 11's opaque BLASes and shader: exactly the pixels the cut-outs are predicted to light must
differ, and no other. Mutation that fails: u and v swapped in the shader (1,766 single-colour pixels wrong, 522 of
1,159 predicted changes).

```
Lanterns stage 23: PASS, alpha-tested shadow rays through non-opaque BLASes, a hit counting only where its barycentrics from image_bvh_intersect_ray give u + 2v < 1: 96750 of 96750 single-colour pixels exact, and against the opaque traversal the cut-outs changed 1159 of 1159 predicted pixels and kept 95449 of 95449
```

## Stage 24: local X and Y thread IDs

Every earlier Lanterns compute shader has a 1-D local size. The title's compute shaders enable both local IDs and read
local Y in v1. `cs_lanterns_xy` has a 16x8 local size (four wave32 waves, `TIDIG_COMP_CNT` 1) on a 2x3 grid of groups
(`TGID_X` and `TGID_Y`). Each lane:

- writes its grid position as gx + 1000 gy (gx = 16 group X + local X, gy = 8 group Y + local Y) to LDS entry
  local Y * 16 + local X;
- meets the group at `s_barrier`, reads the entry of its point mirror in the group (15 - X, 7 - Y), which another wave
  wrote, and stores both words.

How the check works. `lanterns_xy.h` gives both words for every position. All 768 positions must hold exactly their two
words, and the output starts poisoned. Mutation that fails: local X and local Y swapped in the shader (384 of 768 own words
right, 0 mirror words, 768 positions never written).

```
Lanterns stage 24: PASS, a 16x8 local size (four wave32 waves, local X in v0 and local Y in v1) on a 2x3 grid: 768 of 768 positions exact, and 768 of 768 point-mirror words read through LDS after the barrier
```

## Build and run

The guest is opt-in: `-DLEONIDA_TEST_LANTERNS=ON` (off by default). It needs llvm-mc 21 with the AMDGPU target
(`LLVM_MC`, as for `runtime/test/shaders`), spirv-tools, Python 3, and a Vulkan GPU at run time. Default CI builds and
runs none of the GPU part. It does run the CPU tests of the packer and the runner (`tests/test_fssm_pack.py`,
`tests/test_fssm_pack_chain.py`, `tests/test_run_lanterns.py`).

Windows (native):

```sh
cmake -S runtime -B runtime/build -DLEONIDA_TEST_LANTERNS=ON
cmake --build runtime/build --config RelWithDebInfo --target gta6rt lanterns
ctest --test-dir runtime/build -C RelWithDebInfo -R lanterns_stage --output-on-failure
```

Linux with Wine: cross-build with `runtime/cmake/clang-cl-xwin.cmake` as in
[testing.md](testing.md#the-windows-runtime-on-linux-under-wine-or-proton), adding `-DLEONIDA_TEST_LANTERNS=ON`, and
build the targets `gta6rt` and `lanterns`. Then run, in a virtual X server if there is no display:

```sh
B=runtime/build/windows
WINEPREFIX="$HOME/.local/share/leonida-wine" xvfb-run -a python3 -I runtime/test/lanterns/run_lanterns.py --wine \
    --gta6rt $B/gta6rt.exe --guest $B/fixtures/lanterns/RelWithDebInfo/lanterns.exe \
    --shaders $B/generated/lanterns --out $B/lanterns_out --stage 1
```

There is one ctest per stage (`lanterns_stage1`, `lanterns_stage3`, ...; `-R lanterns_stage9` runs one), and the
runner takes any stage in the table with `--stage N`.

The packed containers are in `<build>/generated/lanterns/*.dxbc`. The translated shaders and their index
(`spirv/index.json`, keyed by FSSM SHA-256) are in the same folder. The runner points `LEONIDA_SHADERS` at that
`spirv/` folder.

## Limits

- Tested on one machine so far: an RTX 5080 on Linux, under Wine 11, with the clang-cl/xwin cross-build. No native
  Windows run and no AMD or Intel GPU yet.
- No validation claim. Vulkan validation layers under Wine on that machine are still being looked into, so this page
  does not say the run is validation clean.
- A few frames per stage, offscreen. No present, no frame loop, no swapchain.
- The FSSM programs are synthetic. They use the measured layout (`docs/design.md`, `tools/fssm_header.py`), but only
  the fields the tools read, and the code is ours. A title's programs carry more than this.
- The reference is computed on the CPU, not by another D3D12 implementation.
