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
- Pipelines from DXBC/XDAT/XBCH containers, with each shader found by its FSSM hash.
- Vertex, index and instance buffers in DEFAULT heaps, filled from an upload buffer by `CopyBufferRegion`.
- The vertex shader fetches the vertex and the instance through two V#s in two descriptor tables. It reads the descriptor
  heap start handles through heap slots 9 and 10.
- The model matrix and tint come from 16 root constants, and the camera from a root CBV.
- The floor's pixel shader reads its two checker colours from its own FSSM program, through the code base
  (`s_getpc_b64`).
- The PS input table declares one flat input and one interpolated input.
- Clear, depth, readback through `CopyTextureRegion`, and a fence wait.

How the check works:

- **Exact pixels.** Colours are short binary fractions, so every product the shaders form is exact in float. Pixels
  that are clearly inside one face, or clearly background, must match the CPU reference bit for bit.
- **Uncertain pixels.** A few pixels depend on how the GPU rounds: triangle edges, depth near-ties and checker borders.
  Each of these must be one of the colours the reference lists for it.
- **Lookup control.** Before drawing, the guest builds a container from the vertex shader's FSSM program with one byte
  of its name changed. The layer must refuse that pipeline with `E_INVALIDARG`, because no translated shader has
  that hash. So a pass cannot come from a lookup that matches anything.
- **Negative control.** With the depth test off, submission order decides what is visible. The reference predicts
  which certain pixels change. The GPU frame must change exactly those pixels, and there must be at least 1,000 of
  them. A layer that ignored the depth state, or a reference that could not tell the two frames apart, fails here.

On success the guest prints one line and exits 0:

```
Lanterns stage 1: PASS, 97916 of 98304 pixels exact, 388 edge pixels plausible, depth-off control changed 11601 pixels
```

`run_lanterns.py` passes the run only when all of these hold: exit code 0, this line exactly once, and the shim's
complete factory line, `[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy 0x... (Vulkan device 0x...)`. That last line
means the device is the Vulkan one, not the host's. The runner's timeout kills the whole process group or tree, so a
stalled guest, Wine or GPU fails the run with exit code 3 instead of hanging it. It writes
`lanterns-stage1.png` and `lanterns-stage1-sheet.png`. The sheet shows the GPU frame, the reference, the control
frame, and a mask (grey: exact, red: wrong, yellow: uncertain).

A wrong shader fails. We repacked the floor shader with one checker colour changed (red 0.4375 to 0.5). The guest then
reported 20,430 certain pixels wrong and exited 1, and the runner failed the run.

## Build and run

The guest is opt-in: `-DLEONIDA_TEST_LANTERNS=ON` (off by default). It needs llvm-mc 21 with the AMDGPU target
(`LLVM_MC`, as for `runtime/test/shaders`), spirv-tools, Python 3, and a Vulkan GPU at run time. Default CI builds and
runs none of it.

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
    --shaders $B/generated/lanterns --out lanterns_out --stage 1
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
