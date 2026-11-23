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

## Stage 1

Five `DrawIndexedInstanced` calls draw hexagonal prisms, cubes, octahedra, pyramids and a checker floor into a 384x256
`R32G32B32A32_FLOAT` target with a `D32_FLOAT` depth buffer (depth test LESS). The frame is drawn twice: once with the
depth test (the image) and once without it (the negative control).

What it covers:

- The PE loader, the shims and `D3D12XboxCreateDevice` on Vulkan, called from a mapped guest.
- Pipelines from DXBC/XDAT/XBCH containers, with each shader found by its FSSM hash. The VS container has two 0x0D
  payloads, as 195 of the title's do, and the layer must take the last one, as the tools do.
- Wave32 and wave64 shaders. The octahedra are drawn with a wave64 VS and PS pair; every VS in the title's census is
  wave64. All shaders read their tables from `s0`, where the layer packs root arguments. The title's VS read their
  first pointer at `s8` ([d3d12_vulkan.md](d3d12_vulkan.md) 3.5), but that census does not say which root parameter
  feeds which register, and the layer does not map it, so Lanterns does not test it.
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
Lanterns stage 1: PASS, 97926 of 98304 pixels exact, 378 edge pixels plausible, depth-off control changed 11610 pixels
```

`run_lanterns.py` passes the run only when all of these hold: exit code 0, this line exactly once, whole and with
numbers that fit together (exact and edge pixels add up to the frame, at least 1,000 changed), and the shim's
complete factory line, `[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy 0x... (Vulkan device 0x...)`. That last line
means the device is the Vulkan one, not the host's. The guest waits for each fence for at most 30 s and fails with a
line saying so. With `--wine`, the runner first stops the prefix's `wineserver` (`wineserver -k`, then `-w`,
10 s each): a launch under a new virtual X server while the last run's `explorer.exe` is still alive hangs in
winevulkan's device creation, because that explorer keeps the old, closed display. The runner's timeout kills everything the run started, so a stalled guest, Wine or GPU fails the run
with exit code 3 instead of hanging it. On Linux it also catches a process that left the launcher's session; if
anything survives the kill, or taskkill fails on Windows, the run fails with exit code 4 and names what is left.
Each run writes into a new folder under `--out`: the guest's dumps, the launcher's output, `lanterns-stage1.png`
and `lanterns-stage1-sheet.png`. The sheet shows the GPU frame, the reference, the control frame, and a mask (grey:
exact, red: wrong, yellow: uncertain).

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
- A stale `wineserver`: seen on another Linux setup, where 20 of 20 launches hung without the stop and 20 of 20 passed
  with it. On the test machine two back-to-back runs passed either way, so here it rests on the runner's test with a
  fake `wineserver` on `PATH`.

## Build and run

The guest is opt-in: `-DLEONIDA_TEST_LANTERNS=ON` (off by default). It needs llvm-mc 21 with the AMDGPU target
(`LLVM_MC`, as for `runtime/test/shaders`), spirv-tools, Python 3, and a Vulkan GPU at run time. Default CI builds and
runs none of the GPU part. It does run the CPU tests of the packer and the runner (`tests/test_fssm_pack.py`,
`tests/test_fssm_pack_chain.py`, `tests/test_run_lanterns.py`).

Windows (native):

```sh
cmake -S runtime -B runtime/build -DLEONIDA_TEST_LANTERNS=ON
cmake --build runtime/build --config RelWithDebInfo --target gta6rt lanterns
ctest --test-dir runtime/build -C RelWithDebInfo -R lanterns_stage1 --output-on-failure
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
