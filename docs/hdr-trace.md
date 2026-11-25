# HDR chain trace (opt-in, diagnostics build)

`VICTORY_HDR_TRACE=1` adds a small, one-frame probe of the scene-HDR chain in
`victory-gpu`. It is compiled only with the `diagnostics` feature, reads its
switches once at first use, and records nothing unless all of the following
hold: `VICTORY_HDR_TRACE=1`, `VICTORY_CAPTURE_WORLD` is set, and
`VICTORY_CAPTURE_LISTED_ONLY=1`. Recording is armed for the run (the title
records a frame's commands before presenting it, so filtering by the
presented number misses them); the kept lines are emitted only at the
selected world frame's Present and are discarded at every other Present.
Disabled, it costs one cached boolean and allocates nothing.

## What is recorded, once per frame

1. The first real occurrence of each of these programs, as the host recorded
   it: `34fb51b8a0312e82`, `68029c67dca01425`, `1cd9bc634417a981`,
   `48e74d16cbfa45e3`, `df11dcd012cd7382`, `7c649c422f6208e4`,
   `5970ee6dbc47ec61`.
2. For each selected table slot: static slot and dword offset, SRV/UAV,
   host slot, descriptor address, object and title VA (placed views),
   requested and **observed** effective format, mip/array/plane, generation
   at bind and at submission, stability, retained backing, twin/null.
   "Effective" is the format the host passed to the successful
   `Create*View` for that slot (D3D12 exposes no descriptor read-back). If the slot was remade, the entry is `stable=false` and
   `effective=None` (not observed); nothing is recomputed from a footprint.
   A class mismatch (an F32-expecting slot whose observed view is integer)
   is reported as `class_mismatch=true` only.
3. Two raw-texel copies for the sampled slots and for MRT0 (before and after
   the draw): format of the copied footprint, coordinates and raw bytes,
   with FP16 RGB decoding when the footprint is `R16G16B16A16_FLOAT`. These
   are texels next to the draw, not the value the shader returned.
4. For indirect dispatch/draw: the count (if a count buffer exists) and the
   first 32 argument records, plus the first non-empty record. A missing
   non-empty record in a truncated prefix proves nothing.
5. At `DXGI Present`: present HRESULT, frame and fence, the swap-chain image
   and the source texture presented, and two raw texels of the image.

Sample coordinates are advisory: mid-width at 3/16 and 13/16 of the height
(pixel), or on a 16×16 tile grid, and 16 rows apart for the packed mask
reads. They are not tied to a particular scene feature.

## Launcher variables (diagnostics build, one listed world frame)

```
$env:VICTORY_HDR_TRACE        = '1'
$env:VICTORY_DEPTH_EQUAL_RELAX = '1'
$env:VICTORY_CAPTURE          = 'D:\GTA 6\wt\victory-data\captures\hdr-<stamp>'
$env:VICTORY_CAPTURE_LISTED_ONLY = '1'
$env:VICTORY_CAPTURE_WORLD    = '1'
$env:VICTORY_LOG              = 'info,victory_gpu::hdr_trace=info'
```

Do not set `VICTORY_DUMP_TARGETS`, `VICTORY_DUMP_WRITERS`, `VICTORY_SHOW_TARGET`,
`VICTORY_GPU_TIMING`, `VICTORY_D3D12_DRED` or `VICTORY_D3D12_DEBUG` for this
run. The trace is tied to `VICTORY_CAPTURE_WORLD`, not to
`VICTORY_CAPTURE_FRAMES`; the trace is flushed at the selected world frame's
Present, so commands the title recorded ahead are included.

## Not covered

- No instrumented variant of `34fb`/`6802`, no extra UAV, no new PSO, and no
  claim about the value the shader returned. Depth/stencil and MSAA targets
  are not copied. Only the first occurrence of each program is traced, and
  only the first frame of the list.
- The host does not expose a D3D12 descriptor read-back; the observed format
  is captured at `Create*View` and matched by slot address and generation.
  If the title rewrites the slot after recording, the submission line says
  `stable=false` instead of guessing.
